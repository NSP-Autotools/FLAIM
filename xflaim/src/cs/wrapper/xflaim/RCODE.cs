//------------------------------------------------------------------------------
// Desc:	RCODE
// Tabs:	3
//
//	Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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
	// IMPORTANT NOTE: If the RCODEs in xflaim.h change, those changes need to be reflected here.
	/// <summary>
	/// Error codes returned from XFLAIM.
	/// </summary>
	public enum RCODE : int
	{
		/// <summary></summary>
		NE_XFLM_OK												= 0,
		
		/// <summary></summary>
		NE_XFLM_NOT_IMPLEMENTED								= 0xC05F,
		/// <summary></summary>
		NE_XFLM_MEM												= 0xC037,
		/// <summary></summary>
		NE_XFLM_INVALID_PARM									= 0xC08B,
		/// <summary></summary>
		NE_XFLM_NOT_FOUND										= 0xC006,
		/// <summary></summary>
		NE_XFLM_EXISTS											= 0xC004,
		/// <summary></summary>
		NE_XFLM_FAILURE										= 0xC005,
		/// <summary></summary>
		NE_XFLM_BOF_HIT										= 0xC001,
		/// <summary></summary>
		NE_XFLM_EOF_HIT										= 0xC002,
		/// <summary></summary>
		NE_XFLM_CONV_DEST_OVERFLOW							= 0xC01C,
		/// <summary></summary>
		NE_XFLM_CONV_ILLEGAL									= 0xC01D,
		/// <summary></summary>
		NE_XFLM_CONV_NUM_OVERFLOW							= 0xC020,
		/// <summary></summary>
		NE_XFLM_SYNTAX											= 0xC045,
		/// <summary></summary>
		NE_XFLM_ILLEGAL_OP									= 0xC026,
		/// <summary></summary>
		NE_XFLM_BAD_SEN										= 0xC503,
		/// <summary></summary>
		NE_XFLM_COULD_NOT_START_THREAD					= 0xC504,
		/// <summary></summary>
		NE_XFLM_BAD_BASE64_ENCODING						= 0xC505,
		/// <summary></summary>
		NE_XFLM_STREAM_EXISTS								= 0xC506,
		/// <summary></summary>
		NE_XFLM_MULTIPLE_MATCHES							= 0xC507,
		/// <summary></summary>
		NE_XFLM_NOT_UNIQUE									= 0xC03E,
		/// <summary></summary>
		NE_XFLM_BTREE_ERROR									= 0xC012,
		/// <summary></summary>
		NE_XFLM_BTREE_KEY_SIZE								= 0xC508,
		/// <summary></summary>
		NE_XFLM_BTREE_FULL									= 0xC013,
		/// <summary></summary>
		NE_XFLM_BTREE_BAD_STATE								= 0xC509,
		/// <summary></summary>
		NE_XFLM_COULD_NOT_CREATE_MUTEX					= 0xC50A,
		/// <summary></summary>
		NE_XFLM_DATA_ERROR									= 0xC022,
		/// <summary></summary>
		NE_XFLM_IO_PATH_NOT_FOUND							= 0xC209,
		/// <summary></summary>
		NE_XFLM_IO_END_OF_FILE								= 0xC205,
		/// <summary></summary>
		NE_XFLM_IO_NO_MORE_FILES							= 0xC20C,
		/// <summary></summary>
		NE_XFLM_COULD_NOT_CREATE_SEMAPHORE				= 0xC500,
		/// <summary></summary>
		NE_XFLM_BAD_UTF8										= 0xC501,
		/// <summary></summary>
		NE_XFLM_ERROR_WAITING_ON_SEMPAHORE				= 0xC502,
		/// <summary></summary>
		NE_XFLM_BAD_PLATFORM_FORMAT						= 0xC50B,
			
		/****************************************************************************
		Desc:		General XFLAIM errors
		****************************************************************************/

		/// <summary>User or application aborted (canceled) operation.</summary>
		NE_XFLM_USER_ABORT									= 0xD100,
		/// <summary>Invalid XLM namespace prefix specified.  Either a prefix name or number that was specified was not defined.</summary>
		NE_XFLM_BAD_PREFIX									= 0xD101,
		/// <summary>XML attribute cannot be used - it is being deleted from the database.</summary>
		NE_XFLM_ATTRIBUTE_PURGED							= 0xD102,
		/// <summary>Invalid collection number specified.  Collection is not defined.</summary>
		NE_XFLM_BAD_COLLECTION								= 0xD103,
		/// <summary>Request to lock the database timed out.</summary>
		NE_XFLM_DATABASE_LOCK_REQ_TIMEOUT				= 0xD104,
		/// <summary>Cannot use ELM_ROOT_TAG as a data component in an index.</summary>
		NE_XFLM_ILLEGAL_DATA_COMPONENT					= 0xD105,
		/// <summary>When using ELM_ROOT_TAG in an index component, must specify PRESENCE indexing only.</summary>
		NE_XFLM_MUST_INDEX_ON_PRESENCE					= 0xD106,
		/// <summary>Invalid index number specified.  Index is not defined.</summary>
		NE_XFLM_BAD_IX											= 0xD107,
		/// <summary>Operation could not be performed because a backup is currently in progress.</summary>
		NE_XFLM_BACKUP_ACTIVE								= 0xD108,
		/// <summary>Serial number on backup file does not match the serial number that is expected.</summary>
		NE_XFLM_SERIAL_NUM_MISMATCH						= 0xD109,
		/// <summary>Bad database serial number in roll-forward log file header.</summary>
		NE_XFLM_BAD_RFL_DB_SERIAL_NUM						= 0xD10A,
		/// <summary>Bad roll-forward log file number in roll-forward log file header.</summary>
		NE_XFLM_BAD_RFL_FILE_NUMBER						= 0xD10B,
		/// <summary>Cannot delete an XML element definition in the dictionary because it is in use.</summary>
		NE_XFLM_CANNOT_DEL_ELEMENT							= 0xD10C,
		/// <summary>Cannot modify the data type for an XML element or attribute definition in the dictionary.</summary>
		NE_XFLM_CANNOT_MOD_DATA_TYPE						= 0xD10D,
		/// <summary>Data type of XML element or attribute is not one that can be indexed.</summary>
		NE_XFLM_CANNOT_INDEX_DATA_TYPE					= 0xD10E,
		/// <summary>Bad element number specified - element not defined in dictionary.</summary>
		NE_XFLM_BAD_ELEMENT_NUM								= 0xD10F,
		/// <summary>Bad attribute number specified - attribute not defined in dictionary.</summary>
		NE_XFLM_BAD_ATTRIBUTE_NUM							= 0xD110,
		/// <summary>Bad encryption number specified - encryption definition not defined in dictionary.</summary>
		NE_XFLM_BAD_ENCDEF_NUM								= 0xD111,
		/// <summary>Incremental backup file number provided during a restore is invalid.</summary>
		NE_XFLM_INVALID_FILE_SEQUENCE						= 0xD112,
		/// <summary>Element number specified in element definition is already in use.</summary>
		NE_XFLM_DUPLICATE_ELEMENT_NUM						= 0xD113,
		/// <summary>Illegal transaction type specified for transaction begin operation.</summary>
		NE_XFLM_ILLEGAL_TRANS_TYPE							= 0xD114,
		/// <summary>Version of database found in database header is not supported.</summary>
		NE_XFLM_UNSUPPORTED_VERSION						= 0xD115,
		/// <summary>Illegal operation for transaction type.</summary>
		NE_XFLM_ILLEGAL_TRANS_OP							= 0xD116,
		/// <summary>Incomplete rollback log.</summary>
		NE_XFLM_INCOMPLETE_LOG								= 0xD117,
		/// <summary>Index definition document is illegal - does not conform to the expected form of an index definition document.</summary>
		NE_XFLM_ILLEGAL_INDEX_DEF							= 0xD118,
		/// <summary>The "IndexOn" attribute of an index definition has an illegal value.</summary>
		NE_XFLM_ILLEGAL_INDEX_ON							= 0xD119,
		/// <summary>Attempted an illegal state change on an element or attribute definition.</summary>
		NE_XFLM_ILLEGAL_STATE_CHANGE						= 0xD11A,
		/// <summary>Serial number in roll-forward log file header does not match expected serial number.</summary>
		NE_XFLM_BAD_RFL_SERIAL_NUM							= 0xD11B,
		/// <summary>Running old code on a newer version of database.  Newer code must be used.</summary>
		NE_XFLM_NEWER_FLAIM									= 0xD11C,
		/// <summary>Attempted to change state of a predefined element definition.</summary>
		NE_XFLM_CANNOT_MOD_ELEMENT_STATE					= 0xD11D,
		/// <summary>Attempted to change state of a predefined attribute definition.</summary>
		NE_XFLM_CANNOT_MOD_ATTRIBUTE_STATE				= 0xD11E,
		/// <summary>The highest element number has already been used, cannot create more element definitions.</summary>
		NE_XFLM_NO_MORE_ELEMENT_NUMS						= 0xD11F,
		/// <summary>Operation must be performed inside a database transaction.</summary>
		NE_XFLM_NO_TRANS_ACTIVE								= 0xD120,
		/// <summary>The file specified is not a FLAIM database.</summary>
		NE_XFLM_NOT_FLAIM										= 0xD121,
		/// <summary>Unable to maintain read transaction's view of the database.</summary>
		NE_XFLM_OLD_VIEW										= 0xD122,
		/// <summary>Attempted to perform an operation on the database that requires exclusive access, but cannot because there is a shared lock.</summary>
		NE_XFLM_SHARED_LOCK									= 0xD123,
		/// <summary>Operation cannot be performed while a transaction is active.</summary>
		NE_XFLM_TRANS_ACTIVE									= 0xD124,
		/// <summary>A gap was found in the transaction sequence in the roll-forward log.</summary>
		NE_XFLM_RFL_TRANS_GAP								= 0xD125,
		/// <summary>Something in collated key is bad.</summary>
		NE_XFLM_BAD_COLLATED_KEY							= 0xD126,
		/// <summary>Attempting to delete a collection that has indexes defined for it.  Associated indexes must be deleted before the collection can be deleted.</summary>
		NE_XFLM_MUST_DELETE_INDEXES						= 0xD127,
		/// <summary>Roll-forward log file is incomplete.</summary>
		NE_XFLM_RFL_INCOMPLETE								= 0xD128,
		/// <summary>Cannot restore roll-forward log files - not using multiple roll-forward log files.</summary>
		NE_XFLM_CANNOT_RESTORE_RFL_FILES					= 0xD129,
		/// <summary>A problem (corruption), etc. was detected in a backup set.</summary>
		NE_XFLM_INCONSISTENT_BACKUP						= 0xD12A,
		/// <summary>CRC for database block was invalid.  May indicate problems in reading from or writing to disk.</summary>
		NE_XFLM_BLOCK_CRC										= 0xD12B,
		/// <summary>Attempted operation after a critical error - transaction should be aborted.</summary>
		NE_XFLM_ABORT_TRANS									= 0xD12C,
		/// <summary>File was not a roll-forward log file as expected.</summary>
		NE_XFLM_NOT_RFL										= 0xD12D,
		/// <summary>Roll-forward log file packet was bad.</summary>
		NE_XFLM_BAD_RFL_PACKET								= 0xD12E,
		/// <summary>Bad data path specified to open database.  Does not match data path specified for prior opens of the database.</summary>
		NE_XFLM_DATA_PATH_MISMATCH							= 0xD12F,
		/// <summary>Database must be closed due to a critical error.</summary>
		NE_XFLM_MUST_CLOSE_DATABASE						= 0xD130,
		/// <summary>Encryption key CRC could not be verified.</summary>
		NE_XFLM_INVALID_ENCKEY_CRC							= 0xD131,
		/// <summary>Database header has a bad CRC.</summary>
		NE_XFLM_HDR_CRC										= 0xD132,
		/// <summary>No name table was set up for the database.</summary>
		NE_XFLM_NO_NAME_TABLE								= 0xD133,
		/// <summary>Cannot upgrade database from one version to another.</summary>
		NE_XFLM_UNALLOWED_UPGRADE							= 0xD134,
		/// <summary>Attribute number specified in attribute definition is already in use.</summary>
		NE_XFLM_DUPLICATE_ATTRIBUTE_NUM					= 0xD135,
		/// <summary>Index number specified in index definition is already in use.</summary>
		NE_XFLM_DUPLICATE_INDEX_NUM						= 0xD136,
		/// <summary>Collection number specified in collection definition is already in use.</summary>
		NE_XFLM_DUPLICATE_COLLECTION_NUM					= 0xD137,
		/// <summary>Element name+namespace specified in element definition is already in use.</summary>
		NE_XFLM_DUPLICATE_ELEMENT_NAME					= 0xD138,
		/// <summary>Attribute name+namespace specified in attribute definition is already in use.</summary>
		NE_XFLM_DUPLICATE_ATTRIBUTE_NAME					= 0xD139,
		/// <summary>Index name specified in index definition is already in use.</summary>
		NE_XFLM_DUPLICATE_INDEX_NAME						= 0xD13A,
		/// <summary>Collection name specified in collection definition is already in use.</summary>
		NE_XFLM_DUPLICATE_COLLECTION_NAME				= 0xD13B,
		/// <summary>XML element cannot be used - it is deleted from the database.</summary>
		NE_XFLM_ELEMENT_PURGED								= 0xD13C,
		/// <summary>Too many open databases, cannot open another one.</summary>
		NE_XFLM_TOO_MANY_OPEN_DATABASES					= 0xD13D,
		/// <summary>Operation cannot be performed because the database is currently open.</summary>
		NE_XFLM_DATABASE_OPEN								= 0xD13E,
		/// <summary>Cached database block has been compromised while in cache.</summary>
		NE_XFLM_CACHE_ERROR									= 0xD13F,
		/// <summary>Database is full, cannot create more blocks.</summary>
		NE_XFLM_DB_FULL										= 0xD140,
		/// <summary>Query expression had improper syntax.</summary>
		NE_XFLM_QUERY_SYNTAX									= 0xD141,
		/// <summary>Index is offline, cannot be used in a query.</summary>
		NE_XFLM_INDEX_OFFLINE								= 0xD142,
		/// <summary>Disk which contains roll-forward log is full.</summary>
		NE_XFLM_RFL_DISK_FULL								= 0xD143,
		/// <summary>Must wait for a checkpoint before starting transaction - due to disk problems - usually in disk containing roll-forward log files.</summary>
		NE_XFLM_MUST_WAIT_CHECKPOINT						= 0xD144,
		/// <summary>Encryption definition is missing an encryption algorithm.</summary>
		NE_XFLM_MISSING_ENC_ALGORITHM						= 0xD145,
		/// <summary>Invalid encryption algorithm specified in encryption definition.</summary>
		NE_XFLM_INVALID_ENC_ALGORITHM						= 0xD146,
		/// <summary>Invalid key size specified in encryption definition.</summary>
		NE_XFLM_INVALID_ENC_KEY_SIZE						= 0xD147,
		/// <summary>Data type specified for XML element or attribute definition is illegal.</summary>
		NE_XFLM_ILLEGAL_DATA_TYPE							= 0xD148,
		/// <summary>State specified for index definition or XML element or attribute definition is illegal.</summary>
		NE_XFLM_ILLEGAL_STATE								= 0xD149,
		/// <summary>XML element name specified in element definition is illegal.</summary>
		NE_XFLM_ILLEGAL_ELEMENT_NAME						= 0xD14A,
		/// <summary>XML attribute name specified in attribute definition is illegal.</summary>
		NE_XFLM_ILLEGAL_ATTRIBUTE_NAME					= 0xD14B,
		/// <summary>Collection name specified in collection definition is illegal.</summary>
		NE_XFLM_ILLEGAL_COLLECTION_NAME					= 0xD14C,
		/// <summary>Index name specified is illegal</summary>
		NE_XFLM_ILLEGAL_INDEX_NAME							= 0xD14D,
		/// <summary>Element number specified in element definition or index definition is illegal.</summary>
		NE_XFLM_ILLEGAL_ELEMENT_NUMBER					= 0xD14E,
		/// <summary>Attribute number specified in attribute definition or index definition is illegal.</summary>
		NE_XFLM_ILLEGAL_ATTRIBUTE_NUMBER					= 0xD14F,
		/// <summary>Collection number specified in collection definition or index definition is illegal.</summary>
		NE_XFLM_ILLEGAL_COLLECTION_NUMBER				= 0xD150,
		/// <summary>Index number specified in index definition is illegal.</summary>
		NE_XFLM_ILLEGAL_INDEX_NUMBER						= 0xD151,
		/// <summary>Encryption definition number specified in encryption definition is illegal.</summary>
		NE_XFLM_ILLEGAL_ENCDEF_NUMBER						= 0xD152,
		/// <summary>Collection name and number specified in index definition do not correspond to each other.</summary>
		NE_XFLM_COLLECTION_NAME_MISMATCH					= 0xD153,
		/// <summary>Element name+namespace and number specified in index definition do not correspond to each other.</summary>
		NE_XFLM_ELEMENT_NAME_MISMATCH						= 0xD154,
		/// <summary>Attribute name+namespace and number specified in index definition do not correspond to each other.</summary>
		NE_XFLM_ATTRIBUTE_NAME_MISMATCH					= 0xD155,
		/// <summary>Invalid comparison rule specified in index definition.</summary>
		NE_XFLM_INVALID_COMPARE_RULE						= 0xD156,
		/// <summary>Duplicate key component number specified in index definition.</summary>
		NE_XFLM_DUPLICATE_KEY_COMPONENT					= 0xD157,
		/// <summary>Duplicate data component number specified in index definition.</summary>
		NE_XFLM_DUPLICATE_DATA_COMPONENT					= 0xD158,
		/// <summary>Index definition is missing a key component.</summary>
		NE_XFLM_MISSING_KEY_COMPONENT						= 0xD159,
		/// <summary>Index definition is missing a data component.</summary>
		NE_XFLM_MISSING_DATA_COMPONENT					= 0xD15A,
		/// <summary>Invalid index option specified on index definition.</summary>
		NE_XFLM_INVALID_INDEX_OPTION						= 0xD15B,
		/// <summary>The highest attribute number has already been used, cannot create more.</summary>
		NE_XFLM_NO_MORE_ATTRIBUTE_NUMS					= 0xD15C,
		/// <summary>Missing element name in XML element definition.</summary>
		NE_XFLM_MISSING_ELEMENT_NAME						= 0xD15D,
		/// <summary>Missing attribute name in XML attribute definition.</summary>
		NE_XFLM_MISSING_ATTRIBUTE_NAME					= 0xD15E,
		/// <summary>Missing element number in XML element definition.</summary>
		NE_XFLM_MISSING_ELEMENT_NUMBER					= 0xD15F,
		/// <summary>Missing attribute number from XML attribute definition.</summary>
		NE_XFLM_MISSING_ATTRIBUTE_NUMBER					= 0xD160,
		/// <summary>Missing index name in index definition.</summary>
		NE_XFLM_MISSING_INDEX_NAME							= 0xD161,
		/// <summary>Missing index number in index definition.</summary>
		NE_XFLM_MISSING_INDEX_NUMBER						= 0xD162,
		/// <summary>Missing collection name in collection definition.</summary>
		NE_XFLM_MISSING_COLLECTION_NAME					= 0xD163,
		/// <summary>Missing collection number in collection definition.</summary>
		NE_XFLM_MISSING_COLLECTION_NUMBER				= 0xD164,
		/// <summary>Missing encryption definition name in encryption definition.</summary>
		NE_XFLM_MISSING_ENCDEF_NAME						= 0xD165,
		/// <summary>Missing encryption definition number in encryption definition.</summary>
		NE_XFLM_MISSING_ENCDEF_NUMBER						= 0xD166,
		/// <summary>The highest index number has already been used, cannot create more.</summary>
		NE_XFLM_NO_MORE_INDEX_NUMS							= 0xD167,
		/// <summary>The highest collection number has already been used, cannot create more.</summary>
		NE_XFLM_NO_MORE_COLLECTION_NUMS					= 0xD168,
		/// <summary>Cannot delete an XML attribute definition because it is in use.</summary>
		NE_XFLM_CANNOT_DEL_ATTRIBUTE						= 0xD169,
		/// <summary>Too many documents in the pending document list.</summary>
		NE_XFLM_TOO_MANY_PENDING_NODES					= 0xD16A,
		/// <summary>ELM_ROOT_TAG, if used, must be the sole root component of an index definition.</summary>
		NE_XFLM_BAD_USE_OF_ELM_ROOT_TAG					= 0xD16B,
		/// <summary>Sibling components in an index definition cannot have the same XML element or attribute number.</summary>
		NE_XFLM_DUP_SIBLING_IX_COMPONENTS				= 0xD16C,
		/// <summary>Could not open a roll-forward log file - was not found in the roll-forward log directory.</summary>
		NE_XFLM_RFL_FILE_NOT_FOUND							= 0xD16D,
		/// <summary>Key component of zero in index definition is not allowed.</summary>
		NE_XFLM_ILLEGAL_KEY_COMPONENT_NUM				= 0xD16E,
		/// <summary>Data component of zero in index definition is not allowed.</summary>
		NE_XFLM_ILLEGAL_DATA_COMPONENT_NUM				= 0xD16F,
		/// <summary>Prefix number specified in prefix definition is illegal.</summary>
		NE_XFLM_ILLEGAL_PREFIX_NUMBER						= 0xD170,
		/// <summary>Missing prefix name in prefix definition.</summary>
		NE_XFLM_MISSING_PREFIX_NAME						= 0xD171,
		/// <summary>Missing prefix number in prefix definition.</summary>
		NE_XFLM_MISSING_PREFIX_NUMBER						= 0xD172,
		/// <summary>XML element name+namespace that was specified in index definition or XML document is not defined in dictionary.</summary>
		NE_XFLM_UNDEFINED_ELEMENT_NAME					= 0xD173,
		/// <summary>XML attribute name+namespace that was specified in index definition or XML document is not defined in dictionary.</summary>
		NE_XFLM_UNDEFINED_ATTRIBUTE_NAME					= 0xD174,
		/// <summary>Prefix name specified in prefix definition is already in use.</summary>
		NE_XFLM_DUPLICATE_PREFIX_NAME						= 0xD175,
		/// <summary>Cannot define a namespace for XML attributes whose name begins with "xmlns:" or that is equal to "xmlns"</summary>
		NE_XFLM_NAMESPACE_NOT_ALLOWED						= 0xD176,
		/// <summary>Name for namespace declaration attribute must be "xmlns" or begin with "xmlns:"</summary>
		NE_XFLM_INVALID_NAMESPACE_DECL					= 0xD177,
		/// <summary>Data type for XML attributes that are namespace declarations must be text.</summary>
		NE_XFLM_ILLEGAL_NAMESPACE_DECL_DATATYPE		= 0xD178,
		/// <summary>The highest prefix number has already been used, cannot create more.</summary>
		NE_XFLM_NO_MORE_PREFIX_NUMS						= 0xD179,
		/// <summary>The highest encryption definition number has already been used, cannot create more.</summary>
		NE_XFLM_NO_MORE_ENCDEF_NUMS						= 0xD17A,
		/// <summary>Collection is encrypted, cannot be accessed while in operating in limited mode.</summary>
		NE_XFLM_COLLECTION_OFFLINE							= 0xD17B,
		/// <summary>Item cannot be deleted.</summary>
		NE_XFLM_DELETE_NOT_ALLOWED							= 0xD17C,
		/// <summary>Used during check operations to indicate we need to reset the view.  NOTE: This is an internal error code and should not be documented.</summary>
		NE_XFLM_RESET_NEEDED									= 0xD17D,
		/// <summary>An illegal value was specified for the "Required" attribute in an index definition.</summary>
		NE_XFLM_ILLEGAL_REQUIRED_VALUE					= 0xD17E,
		/// <summary>A leaf index component in an index definition was not marked as a data component or key component.</summary>
		NE_XFLM_ILLEGAL_INDEX_COMPONENT					= 0xD17F,
		/// <summary>Illegal value for the "UniqueSubElements" attribute in an element definition.</summary>
		NE_XFLM_ILLEGAL_UNIQUE_SUB_ELEMENT_VALUE		= 0xD180,
		/// <summary>Data type for an element definition with UniqueSubElements="yes" must be nodata.</summary>
		NE_XFLM_DATA_TYPE_MUST_BE_NO_DATA				= 0xD181,
		/// <summary>Cannot set the "Required" attribute on a non-key index component in index definition.</summary>
		NE_XFLM_CANNOT_SET_REQUIRED						= 0xD182,
		/// <summary>Cannot set the "Limit" attribute on a non-key index component in index definition.</summary>
		NE_XFLM_CANNOT_SET_LIMIT							= 0xD183,
		/// <summary>Cannot set the "IndexOn" attribute on a non-key index component in index definition.</summary>
		NE_XFLM_CANNOT_SET_INDEX_ON						= 0xD184,
		/// <summary>Cannot set the "CompareRules" on a non-key index component in index definition.</summary>
		NE_XFLM_CANNOT_SET_COMPARE_RULES					= 0xD185,
		/// <summary>Attempt to set a value while an input stream is still open.</summary>
		NE_XFLM_INPUT_PENDING								= 0xD186,
		/// <summary>Bad node type</summary>
		NE_XFLM_INVALID_NODE_TYPE							= 0xD187,
		/// <summary>Attempt to insert a unique child element that has a lower node ID than the parent element</summary>
		NE_XFLM_INVALID_CHILD_ELM_NODE_ID				= 0xD188,
		/// <summary>Hit the end of the RFL</summary>
		NE_XFLM_RFL_END										= 0xD189,
		/// <summary>Illegal flag passed to getChildElement method.  Must be zero for elements that can have non-unique child elements.</summary>
		NE_XFLM_ILLEGAL_FLAG									= 0xD18A,
		/// <summary>Operation timed out.</summary>
		NE_XFLM_TIMEOUT										= 0xD18B,
		/// <summary>Non-numeric digit found in text to numeric conversion.</summary>
		NE_XFLM_CONV_BAD_DIGIT								= 0xD18C,
		/// <summary>Data source cannot be NULL when doing data conversion.</summary>
		NE_XFLM_CONV_NULL_SRC								= 0xD18D,
		/// <summary>Numeric underflow (&lt; lower bound) converting to numeric type.</summary>
		NE_XFLM_CONV_NUM_UNDERFLOW							= 0xD18E,
		/// <summary>Attempting to use a feature for which full support has been disabled.</summary>
		NE_XFLM_UNSUPPORTED_FEATURE						= 0xD18F,
		/// <summary>Attempt to create a database, but the file already exists.</summary>
		NE_XFLM_FILE_EXISTS									= 0xD190,
		/// <summary>Buffer overflow.</summary>
		NE_XFLM_BUFFER_OVERFLOW								= 0xD191,
		/// <summary>Invalid XML encountered while parsing document.</summary>
		NE_XFLM_INVALID_XML									= 0xD192,
		/// <summary>Attempt to set/get data on an XML element or attribute using a data type that is incompatible with the data type specified in the dictionary.</summary>
		NE_XFLM_BAD_DATA_TYPE								= 0xD193,
		/// <summary>Item is read-only and cannot be updated.</summary>
		NE_XFLM_READ_ONLY										= 0xD194,
		/// <summary>Generated index key too large.</summary>
		NE_XFLM_KEY_OVERFLOW									= 0xD195,
		/// <summary>Encountered unexpected end of input when parsing XPATH expression.</summary>
		NE_XFLM_UNEXPECTED_END_OF_INPUT					= 0xD196,
			
		/****************************************************************************
		Desc:		DOM Errors
		****************************************************************************/

		/// <summary>Attempt to insert a DOM node somewhere it doesn't belong.</summary>
		NE_XFLM_DOM_HIERARCHY_REQUEST_ERR				= 0xD201,
		/// <summary>A DOM node is being used in a different document than the one that created it.</summary>
		NE_XFLM_DOM_WRONG_DOCUMENT_ERR					= 0xD202,
		/// <summary>Links between DOM nodes in a document are corrupt.</summary>
		NE_XFLM_DOM_DATA_ERROR								= 0xD203,
		/// <summary>The requested DOM node does not exist.</summary>
		NE_XFLM_DOM_NODE_NOT_FOUND							= 0xD204,
		/// <summary>Attempting to insert a child DOM node whose type cannot be inserted as a child node.</summary>
		NE_XFLM_DOM_INVALID_CHILD_TYPE					= 0xD205,
		/// <summary>DOM node being accessed has been deleted.</summary>
		NE_XFLM_DOM_NODE_DELETED							= 0xD206,
		/// <summary>Node already has a child element with the given name id - this node's child nodes must all be unique.</summary>
		NE_XFLM_DOM_DUPLICATE_ELEMENT						= 0xD207,

		/****************************************************************************
		Desc:	Query Errors
		****************************************************************************/

		/// <summary>Query setup error: Unmatched right paren.</summary>
		NE_XFLM_Q_UNMATCHED_RPAREN							= 0xD301,
		/// <summary>Query setup error: Unexpected left paren.</summary>
		NE_XFLM_Q_UNEXPECTED_LPAREN						= 0xD302,
		/// <summary>Query setup error: Unexpected right paren.</summary>
		NE_XFLM_Q_UNEXPECTED_RPAREN						= 0xD303,
		/// <summary>Query setup error: Expecting an operand.</summary>
		NE_XFLM_Q_EXPECTING_OPERAND						= 0xD304,
		/// <summary>Query setup error: Expecting an operator.</summary>
		NE_XFLM_Q_EXPECTING_OPERATOR						= 0xD305,
		/// <summary>Query setup error: Unexpected comma.</summary>
		NE_XFLM_Q_UNEXPECTED_COMMA							= 0xD306,
		/// <summary>Query setup error: Expecting a left paren.</summary>
		NE_XFLM_Q_EXPECTING_LPAREN							= 0xD307,
		/// <summary>Query setup error: Unexpected value.</summary>
		NE_XFLM_Q_UNEXPECTED_VALUE							= 0xD308,
		/// <summary>Query setup error: Invalid number of arguments for a function.</summary>
		NE_XFLM_Q_INVALID_NUM_FUNC_ARGS					= 0xD309,
		/// <summary>Query setup error: Unexpected XPATH componenent.</summary>
		NE_XFLM_Q_UNEXPECTED_XPATH_COMPONENT			= 0xD30A,
		/// <summary>Query setup error: Illegal left bracket ([).</summary>
		NE_XFLM_Q_ILLEGAL_LBRACKET							= 0xD30B,
		/// <summary>Query setup error: Illegal right bracket (]).</summary>
		NE_XFLM_Q_ILLEGAL_RBRACKET							= 0xD30C,
		/// <summary>Query setup error: Operand for some operator is not valid for that operator type.</summary>
		NE_XFLM_Q_ILLEGAL_OPERAND							= 0xD30D,
		/// <summary>Operation is illegal, cannot change certain things after query has been optimized.</summary>
		NE_XFLM_Q_ALREADY_OPTIMIZED						= 0xD30E,
		/// <summary>Database handle passed in does not match database associated with query.</summary>
		NE_XFLM_Q_MISMATCHED_DB								= 0xD30F,
		/// <summary>Illegal operator - cannot pass this operator into the addOperator method.</summary>
		NE_XFLM_Q_ILLEGAL_OPERATOR							= 0xD310,
		/// <summary>Illegal combination of comparison rules passed to addOperator method.</summary>
		NE_XFLM_Q_ILLEGAL_COMPARE_RULES					= 0xD311,
		/// <summary>Query setup error: Query expression is incomplete.</summary>
		NE_XFLM_Q_INCOMPLETE_QUERY_EXPR					= 0xD312,
		/// <summary>Query not positioned due to previous error, cannot call getNext, getPrev, or getCurrent</summary>
		NE_XFLM_Q_NOT_POSITIONED							= 0xD313,
		/// <summary>Query setup error: Invalid type of value constant used for node id value comparison.</summary>
		NE_XFLM_Q_INVALID_NODE_ID_VALUE					= 0xD314,
		/// <summary>Query setup error: Invalid meta data type specified.</summary>
		NE_XFLM_Q_INVALID_META_DATA_TYPE					= 0xD315,
		/// <summary>Query setup error: Cannot add an expression to an XPATH component after having added an expression that tests context position.</summary>
		NE_XFLM_Q_NEW_EXPR_NOT_ALLOWED					= 0xD316,
		/// <summary>Invalid context position value encountered - must be a positive number.</summary>
		NE_XFLM_Q_INVALID_CONTEXT_POS						= 0xD317,
		/// <summary>Query setup error: Parameter to user-defined functions must be a single XPATH only.</summary>
		NE_XFLM_Q_INVALID_FUNC_ARG							= 0xD318,
		/// <summary>Query setup error: Expecting right paren.</summary>
		NE_XFLM_Q_EXPECTING_RPAREN							= 0xD319,
		/// <summary>Query setup error: Cannot add sort keys after having called getFirst, getLast, getNext, or getPrev.</summary>
		NE_XFLM_Q_TOO_LATE_TO_ADD_SORT_KEYS				= 0xD31A,
		/// <summary>Query setup error: Invalid sort key component number specified in query.</summary>
		NE_XFLM_Q_INVALID_SORT_KEY_COMPONENT			= 0xD31B,
		/// <summary>Query setup error: Duplicate sort key component number specified in query.</summary>
		NE_XFLM_Q_DUPLICATE_SORT_KEY_COMPONENT			= 0xD31C,
		/// <summary>Query setup error: Missing sort key component number in sort keys that were specified for query.</summary>
		NE_XFLM_Q_MISSING_SORT_KEY_COMPONENT			= 0xD31D,
		/// <summary>Query setup error: addSortKeys was called, but no sort key components were specified.</summary>
		NE_XFLM_Q_NO_SORT_KEY_COMPONENTS_SPECIFIED	= 0xD31E,
		/// <summary>Query setup error: A sort key context cannot be an XML attribute.</summary>
		NE_XFLM_Q_SORT_KEY_CONTEXT_MUST_BE_ELEMENT	= 0xD31F,
		/// <summary>Query setup error: The XML element number specified for a sort key in a query is invalid - no element definition in the dictionary.</summary>
		NE_XFLM_Q_INVALID_ELEMENT_NUM_IN_SORT_KEYS	= 0xD320,
		/// <summary>Query setup error: The XML attribute number specified for a sort key in a query is invalid - no attribute definition in the dictionary.</summary>
		NE_XFLM_Q_INVALID_ATTR_NUM_IN_SORT_KEYS		= 0xD321,
		/// <summary>Attempt is being made to position in a query that is not positionable.</summary>
		NE_XFLM_Q_NON_POSITIONABLE_QUERY					= 0xD322,
		/// <summary>Attempt is being made to position to an invalid position in the result set.</summary>
		NE_XFLM_Q_INVALID_POSITION							= 0xD323,

		/****************************************************************************
		Desc:	NICI / Encryption Errors
		****************************************************************************/

		/// <summary>Error occurred while creating NICI context for encryption/decryption.</summary>
		NE_XFLM_NICI_CONTEXT									= 0xD401,
		/// <summary>Error occurred while accessing an attribute on a NICI encryption key.</summary>
		NE_XFLM_NICI_ATTRIBUTE_VALUE						= 0xD402,
		/// <summary>Value retrieved from an attribute on a NICI encryption key was bad.</summary>
		NE_XFLM_NICI_BAD_ATTRIBUTE							= 0xD403,
		/// <summary>Error occurred while wrapping a NICI encryption key in another NICI encryption key.</summary>
		NE_XFLM_NICI_WRAPKEY_FAILED						= 0xD404,
		/// <summary>Error occurred while unwrapping a NICI encryption key that is wrapped in another NICI encryption key.</summary>
		NE_XFLM_NICI_UNWRAPKEY_FAILED						= 0xD405,
		/// <summary>Attempt to use invalid NICI encryption algorithm. </summary>
		NE_XFLM_NICI_INVALID_ALGORITHM					= 0xD406,
		/// <summary>Error occurred while attempting to generate a NICI encryption key.</summary>
		NE_XFLM_NICI_GENKEY_FAILED							= 0xD407,
		/// <summary>Error occurred while generating random data using NICI.</summary>
		NE_XFLM_NICI_BAD_RANDOM								= 0xD408,
		/// <summary>Error occurred while attempting to wrap a NICI encryption key in a password.</summary>
		NE_XFLM_PBE_ENCRYPT_FAILED							= 0xD409,
		/// <summary>Error occurred while attempting to unwrap a NICI encryption key that was previously wrapped in a password.</summary>
		NE_XFLM_PBE_DECRYPT_FAILED							= 0xD40A,
		/// <summary>Error occurred while attempting to initialize the NICI digest functionality.</summary>
		NE_XFLM_DIGEST_INIT_FAILED							= 0xD40B,
		/// <summary>Error occurred while attempting to create a NICI digest. </summary>
		NE_XFLM_DIGEST_FAILED								= 0xD40C,
		/// <summary>Error occurred while attempting to inject an encryption key into NICI. </summary>
		NE_XFLM_INJECT_KEY_FAILED							= 0xD40D,
		/// <summary>Error occurred while attempting to initialize NICI to find information on a NICI encryption key.</summary>
		NE_XFLM_NICI_FIND_INIT								= 0xD40E,
		/// <summary>Error occurred while attempting to find information on a NICI encryption key.</summary>
		NE_XFLM_NICI_FIND_OBJECT							= 0xD40F,
		/// <summary>Could not find the NICI encryption key or information on the NICI encryption key.</summary>
		NE_XFLM_NICI_KEY_NOT_FOUND							= 0xD410,
		/// <summary>Error occurred while initializing NICI to encrypt data.</summary>
		NE_XFLM_NICI_ENC_INIT_FAILED						= 0xD411,
		/// <summary>Error occurred while encrypting data.</summary>
		NE_XFLM_NICI_ENCRYPT_FAILED						= 0xD412,
		/// <summary>Error occurred while initializing NICI to decrypt data.</summary>
		NE_XFLM_NICI_DECRYPT_INIT_FAILED					= 0xD413,
		/// <summary>Error occurred while decrypting data.</summary>
		NE_XFLM_NICI_DECRYPT_FAILED						= 0xD414,
		/// <summary>Could not find the NICI encryption key used to wrap another NICI encryption key.</summary>
		NE_XFLM_NICI_WRAPKEY_NOT_FOUND					= 0xD415,
		/// <summary>Password supplied when none was expected.</summary>
		NE_XFLM_NOT_EXPECTING_PASSWORD					= 0xD416,
		/// <summary>No password supplied when one was required.</summary>
		NE_XFLM_EXPECTING_PASSWORD							= 0xD417,
		/// <summary>Error occurred while attempting to extract a NICI encryption key.</summary>
		NE_XFLM_EXTRACT_KEY_FAILED							= 0xD418,
		/// <summary>Error occurred while initializing NICI.</summary>
		NE_XFLM_NICI_INIT_FAILED							= 0xD419,
		/// <summary>Bad encryption key size found in roll-forward log packet.</summary>
		NE_XFLM_BAD_ENCKEY_SIZE								= 0xD41A,
		/// <summary>Attempt was made to encrypt data when NICI is unavailable.</summary>
		NE_XFLM_ENCRYPTION_UNAVAILABLE					= 0xD41B,

		/****************************************************************************
		Desc:	Toolkit errors
		****************************************************************************/

		/// <summary>Beginning of results encountered.</summary>
		NE_FLM_BOF_HIT											= 0xC001,
		/// <summary>End of results encountered.</summary>
		NE_FLM_EOF_HIT											= 0xC002,
		/// <summary>Object already exists.</summary>
		NE_FLM_EXISTS											= 0xC004,
		/// <summary>Internal failure.</summary>
		NE_FLM_FAILURE											= 0xC005,
		/// <summary>An object was not found.</summary>
		NE_FLM_NOT_FOUND										= 0xC006,
		/// <summary>Corruption found in b-tree.</summary>
		NE_FLM_BTREE_ERROR									= 0xC012,
		/// <summary>B-tree cannot grow beyond current size.</summary>
		NE_FLM_BTREE_FULL										= 0xC013,
		/// <summary>Destination buffer not large enough to hold data.</summary>
		NE_FLM_CONV_DEST_OVERFLOW							= 0xC01C,
		/// <summary>Attempt to convert between data types is an unsupported conversion.</summary>
		NE_FLM_CONV_ILLEGAL									= 0xC01D,
		/// <summary>Numeric overflow (> upper bound) converting to numeric type.</summary>
		NE_FLM_CONV_NUM_OVERFLOW							= 0xC020,
		/// <summary>Corruption found in b-tree.</summary>
		NE_FLM_DATA_ERROR										= 0xC022,
		/// <summary>Illegal operation</summary>
		NE_FLM_ILLEGAL_OP										= 0xC026,
		/// <summary>Attempt to allocate memory failed.</summary>
		NE_FLM_MEM												= 0xC037,
		/// <summary>Non-unique key.</summary>
		NE_FLM_NOT_UNIQUE										= 0xC03E,
		/// <summary>Syntax error while parsing.</summary>
		NE_FLM_SYNTAX											= 0xC045,
		/// <summary>Attempt was made to use a feature that is not implemented.</summary>
		NE_FLM_NOT_IMPLEMENTED								= 0xC05F,
		/// <summary>Invalid parameter passed into a function.</summary>
		NE_FLM_INVALID_PARM									= 0xC08B,

		// I/O Errors - Must be the same as they were for FLAIM.

		/// <summary>Access to file is denied.  Caller is not allowed access to a file.</summary>
		NE_FLM_IO_ACCESS_DENIED								= 0xC201,
		/// <summary>Bad file handle or file descriptor.</summary>
		NE_FLM_IO_BAD_FILE_HANDLE							= 0xC202,
		/// <summary>Error occurred while copying a file.</summary>
		NE_FLM_IO_COPY_ERR									= 0xC203,
		/// <summary>Disk full.</summary>
		NE_FLM_IO_DISK_FULL									= 0xC204,
		/// <summary>End of file reached while reading from the file.</summary>
		NE_FLM_IO_END_OF_FILE								= 0xC205,
		/// <summary>Error while opening the file.</summary>
		NE_FLM_IO_OPEN_ERR									= 0xC206,
		/// <summary>Error occurred while positioning (seeking) within a file.</summary>
		NE_FLM_IO_SEEK_ERR									= 0xC207,
		/// <summary>Error occurred while accessing or deleting a directory.</summary>
		NE_FLM_IO_DIRECTORY_ERR								= 0xC208,
		/// <summary>File not found.</summary>
		NE_FLM_IO_PATH_NOT_FOUND							= 0xC209,
		/// <summary>Too many files open.</summary>
		NE_FLM_IO_TOO_MANY_OPEN_FILES						= 0xC20A,
		/// <summary>File name too long.</summary>
		NE_FLM_IO_PATH_TOO_LONG								= 0xC20B,
		/// <summary>No more files in directory.</summary>
		NE_FLM_IO_NO_MORE_FILES								= 0xC20C,
		/// <summary>Error occurred while deleting a file.</summary>
		NE_FLM_IO_DELETING_FILE								= 0xC20D,
		/// <summary>Error attempting to acquire a byte-range lock on a file.</summary>
		NE_FLM_IO_FILE_LOCK_ERR								= 0xC20E,
		/// <summary>Error attempting to release a byte-range lock on a file.</summary>
		NE_FLM_IO_FILE_UNLOCK_ERR							= 0xC20F,
		/// <summary>Error occurred while attempting to create a directory or sub-directory.</summary>
		NE_FLM_IO_PATH_CREATE_FAILURE						= 0xC210,
		/// <summary>Error occurred while renaming a file.</summary>
		NE_FLM_IO_RENAME_FAILURE							= 0xC211,
		/// <summary>Invalid file password.</summary>
		NE_FLM_IO_INVALID_PASSWORD							= 0xC212,
		/// <summary>Error occurred while setting up to perform a file read operation.</summary>
		NE_FLM_SETTING_UP_FOR_READ							= 0xC213,
		/// <summary>Error occurred while setting up to perform a file write operation.</summary>
		NE_FLM_SETTING_UP_FOR_WRITE						= 0xC214,
		/// <summary>Cannot reduce file name into more components.</summary>
		NE_FLM_IO_CANNOT_REDUCE_PATH						= 0xC215,
		/// <summary>Error occurred while setting up to access the file system.</summary>
		NE_FLM_INITIALIZING_IO_SYSTEM						= 0xC216,
		/// <summary>Error occurred while flushing file data buffers to disk.</summary>
		NE_FLM_FLUSHING_FILE									= 0xC217,
		/// <summary>Invalid file name.</summary>
		NE_FLM_IO_INVALID_FILENAME							= 0xC218,
		/// <summary>Error connecting to a remote network resource.</summary>
		NE_FLM_IO_CONNECT_ERROR								= 0xC219,
		/// <summary>Unexpected error occurred while opening a file.</summary>
		NE_FLM_OPENING_FILE									= 0xC21A,
		/// <summary>Unexpected error occurred while opening a file in direct access mode.</summary>
		NE_FLM_DIRECT_OPENING_FILE							= 0xC21B,
		/// <summary>Unexpected error occurred while creating a file.</summary>
		NE_FLM_CREATING_FILE									= 0xC21C,
		/// <summary>Unexpected error occurred while creating a file in direct access mode.</summary>
		NE_FLM_DIRECT_CREATING_FILE						= 0xC21D,
		/// <summary>Unexpected error occurred while reading a file.</summary>
		NE_FLM_READING_FILE									= 0xC21E,
		/// <summary>Unexpected error occurred while reading a file in direct access mode.</summary>
		NE_FLM_DIRECT_READING_FILE							= 0xC21F,
		/// <summary>Unexpected error occurred while writing to a file.</summary>
		NE_FLM_WRITING_FILE									= 0xC220,
		/// <summary>Unexpected error occurred while writing a file in direct access mode.</summary>
		NE_FLM_DIRECT_WRITING_FILE							= 0xC221,
		/// <summary>Unexpected error occurred while positioning within a file.</summary>
		NE_FLM_POSITIONING_IN_FILE							= 0xC222,
		/// <summary>Unexpected error occurred while getting a file's size.</summary>
		NE_FLM_GETTING_FILE_SIZE							= 0xC223,
		/// <summary>Unexpected error occurred while truncating a file.</summary>
		NE_FLM_TRUNCATING_FILE								= 0xC224,
		/// <summary>Unexpected error occurred while parsing a file's name.</summary>
		NE_FLM_PARSING_FILE_NAME							= 0xC225,
		/// <summary>Unexpected error occurred while closing a file.</summary>
		NE_FLM_CLOSING_FILE									= 0xC226,
		/// <summary>Unexpected error occurred while getting information about a file.</summary>
		NE_FLM_GETTING_FILE_INFO							= 0xC227,
		/// <summary>Unexpected error occurred while expanding a file.</summary>
		NE_FLM_EXPANDING_FILE								= 0xC228,
		/// <summary>Unexpected error getting free blocks from file system.</summary>
		NE_FLM_GETTING_FREE_BLOCKS							= 0xC229,
		/// <summary>Unexpected error occurred while checking to see if a file exists.</summary>
		NE_FLM_CHECKING_FILE_EXISTENCE					= 0xC22A,
		/// <summary>Unexpected error occurred while renaming a file.</summary>
		NE_FLM_RENAMING_FILE									= 0xC22B,
		/// <summary>Unexpected error occurred while setting a file's information.</summary>
		NE_FLM_SETTING_FILE_INFO							= 0xC22C,
		/// <summary>I/O has not yet completed</summary>
		NE_FLM_IO_PENDING										= 0xC22D,
		/// <summary>An async I/O operation failed</summary>
		NE_FLM_ASYNC_FAILED									= 0xC22E,
		/// <summary>Misaligned buffer or offset encountered during I/O request</summary>
		NE_FLM_MISALIGNED_IO									= 0xC22F,
			
		// Stream Errors - These are new

		/// <summary>Error decompressing data stream.</summary>
		NE_FLM_STREAM_DECOMPRESS_ERROR					= 0xC400,
		/// <summary>Attempting to decompress a data stream that is not compressed.</summary>
		NE_FLM_STREAM_NOT_COMPRESSED						= 0xC401,
		/// <summary>Too many files in input stream.</summary>
		NE_FLM_STREAM_TOO_MANY_FILES						= 0xC402,
			
		// Miscellaneous new toolkit errors
			
		/// <summary>Could not create a semaphore.</summary>
		NE_FLM_COULD_NOT_CREATE_SEMAPHORE				= 0xC500,
		/// <summary>An invalid byte sequence was found in a UTF-8 string</summary>
		NE_FLM_BAD_UTF8										= 0xC501,
		/// <summary>Error occurred while waiting on a sempahore.</summary>
		NE_FLM_ERROR_WAITING_ON_SEMAPHORE				= 0xC502,
		/// <summary>Invalid simple encoded number.</summary>
		NE_FLM_BAD_SEN											= 0xC503,
		/// <summary>Problem starting a new thread.</summary>
		NE_FLM_COULD_NOT_START_THREAD						= 0xC504,
		/// <summary>Invalid base64 sequence encountered.</summary>
		NE_FLM_BAD_BASE64_ENCODING							= 0xC505,
		/// <summary>Stream file already exists.</summary>
		NE_FLM_STREAM_EXISTS									= 0xC506,
		/// <summary>Multiple items matched but only one match was expected.</summary>
		NE_FLM_MULTIPLE_MATCHES								= 0xC507,
		/// <summary>Invalid b-tree key size.</summary>
		NE_FLM_BTREE_KEY_SIZE								= 0xC508,
		/// <summary>B-tree operation cannot be completed.</summary>
		NE_FLM_BTREE_BAD_STATE								= 0xC509,
		/// <summary>Error occurred while creating or initializing a mutex.</summary>
		NE_FLM_COULD_NOT_CREATE_MUTEX						= 0xC50A,
		/// <summary>In-memory alignment of disk structures is incorrect</summary>
		NE_FLM_BAD_PLATFORM_FORMAT							= 0xC50B,
		/// <summary>Timeout while waiting for a lock object</summary>
		NE_FLM_LOCK_REQ_TIMEOUT								= 0xC50C,
		/// <summary>Timeout while waiting on a semaphore, condition variable, or reader/writer lock</summary>
		NE_FLM_WAIT_TIMEOUT									= 0xC50D,
			
		// Network Errors - Must be the same as they were for FLAIM

		/// <summary>IP address not found</summary>
		NE_FLM_NOIP_ADDR										= 0xC900,
		/// <summary>IP socket failure</summary>
		NE_FLM_SOCKET_FAIL									= 0xC901,
		/// <summary>TCP/IP connection failure</summary>
		NE_FLM_CONNECT_FAIL									= 0xC902,
		/// <summary>The TCP/IP services on your system may not be configured or installed.</summary>
		NE_FLM_BIND_FAIL										= 0xC903,
		/// <summary>TCP/IP listen failed</summary>
		NE_FLM_LISTEN_FAIL									= 0xC904,
		/// <summary>TCP/IP accept failed</summary>
		NE_FLM_ACCEPT_FAIL									= 0xC905,
		/// <summary>TCP/IP select failed</summary>
		NE_FLM_SELECT_ERR										= 0xC906,
		/// <summary>TCP/IP socket operation failed</summary>
		NE_FLM_SOCKET_SET_OPT_FAIL							= 0xC907,
		/// <summary>TCP/IP disconnected</summary>
		NE_FLM_SOCKET_DISCONNECT							= 0xC908,
		/// <summary>TCP/IP read failed</summary>
		NE_FLM_SOCKET_READ_FAIL								= 0xC909,
		/// <summary>TCP/IP write failed</summary>
		NE_FLM_SOCKET_WRITE_FAIL							= 0xC90A,
		/// <summary>TCP/IP read timeout</summary>
		NE_FLM_SOCKET_READ_TIMEOUT							= 0xC90B,
		/// <summary>TCP/IP write timeout</summary>
		NE_FLM_SOCKET_WRITE_TIMEOUT						= 0xC90C,
		/// <summary>Connection already closed</summary>
		NE_FLM_SOCKET_ALREADY_CLOSED						= 0xC90D,
	}
}
