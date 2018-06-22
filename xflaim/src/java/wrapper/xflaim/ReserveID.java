//------------------------------------------------------------------------------
// Desc:	Reserve ID
// Tabs:	3
//
// Copyright (c) 2003, 2005-2007 Novell, Inc. All Rights Reserved.
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
 * This class provides enums for all of the reserved dictionay tags in XFlaim. 
 */
public final class ReserveID
{
	public static final int ELM_ELEMENT_TAG								= 0xFFFFFE00;
	public static final String ELM_ELEMENT_TAG_NAME						= "element";
	public static final int ELM_ATTRIBUTE_TAG								= 0xFFFFFE01;
	public static final String ELM_ATTRIBUTE_TAG_NAME					= "attribute";
	public static final int ELM_INDEX_TAG									= 0xFFFFFE02;
	public static final String ELM_INDEX_TAG_NAME						= "Index";
	public static final int ELM_ELEMENT_COMPONENT_TAG					= 0xFFFFFE04;
	public static final String ELM_ELEMENT_COMPONENT_TAG_NAME		= "ElementComponent";
	public static final int ELM_ATTRIBUTE_COMPONENT_TAG				= 0xFFFFFE05;
	public static final String ELM_ATTRIBUTE_COMPONENT_TAG_NAME		= "AttributeComponent";
	public static final int ELM_COLLECTION_TAG							= 0xFFFFFE06;
	public static final String ELM_COLLECTION_TAG_NAME					= "Collection";
	public static final int ELM_PREFIX_TAG									= 0xFFFFFE07;
	public static final String ELM_PREFIX_TAG_NAME						= "Prefix";
	public static final int ELM_NEXT_DICT_NUMS_TAG						= 0xFFFFFE08;
	public static final String ELM_NEXT_DICT_NUMS_TAG_NAME			= "NextDictNums";
	public static final int ELM_DOCUMENT_TITLE_TAG						= 0xFFFFFE09;
	public static final String ELM_DOCUMENT_TITLE_TAG_NAME			= "DocumentTitle";
	public static final int ELM_INVALID_TAG								= 0xFFFFFE0A;
	public static final String ELM_INVALID_TAG_NAME						= "Invalid";
	public static final int ELM_QUARANTINED_TAG							= 0xFFFFFE0B;
	public static final String ELM_QUARANTINED_TAG_NAME				= "Quarantined";
	public static final int ELM_ALL_TAG										= 0xFFFFFE0C;
	public static final String ELM_ALL_TAG_NAME							= "All";
	public static final int ELM_ANNOTATION_TAG							= 0xFFFFFE0D;
	public static final String ELM_ANNOTATION_TAG_NAME					= "Annotation";
	public static final int ELM_ANY_TAG										= 0xFFFFFE0E;
	public static final String ELM_ANY_TAG_NAME							= "Any";
	public static final int ELM_ATTRIBUTE_GROUP_TAG						= 0xFFFFFE0F;
	public static final String ELM_ATTRIBUTE_GROUP_TAG_NAME			= "AttributeGroup";
	public static final int ELM_CHOICE_TAG									= 0xFFFFFE10;
	public static final String ELM_CHOICE_TAG_NAME						= "Choice";
	public static final int ELM_COMPLEX_CONTENT_TAG						= 0xFFFFFE11;
	public static final String ELM_COMPLEX_CONTENT_TAG_NAME			= "ComplexContent";
	public static final int ELM_COMPLEX_TYPE_TAG							= 0xFFFFFE12;
	public static final String ELM_COMPLEX_TYPE_TAG_NAME				= "ComplexType";
	public static final int ELM_DOCUMENTATION_TAG						= 0xFFFFFE13;
	public static final String ELM_DOCUMENTATION_TAG_NAME				= "Documentation";
	public static final int ELM_ENUMERATION_TAG							= 0xFFFFFE14;
	public static final String ELM_ENUMERATION_TAG_NAME				= "Enumeration";
	public static final int ELM_EXTENSION_TAG								= 0xFFFFFE15;
	public static final String ELM_EXTENSION_TAG_NAME					= "Extension";
	public static final int ELM_GROUP_TAG									= 0xFFFFFE16;
	public static final String ELM_GROUP_TAG_NAME						= "Group";
	public static final int ELM_MAX_OCCURS_TAG							= 0xFFFFFE17;
	public static final String ELM_MAX_OCCURS_TAG_NAME					= "MaxOccurs";
	public static final int ELM_MIN_OCCURS_TAG							= 0xFFFFFE18;
	public static final String ELM_MIN_OCCURS_TAG_NAME					= "MinOccurs";
	public static final int ELM_RESTRICTION_TAG							= 0xFFFFFE19;
	public static final String ELM_RESTRICTION_TAG_NAME				= "Restriction";
	public static final int ELM_SEQUENCE_TAG								= 0xFFFFFE1A;
	public static final String ELM_SEQUENCE_TAG_NAME					= "Sequence";
	public static final int ELM_SIMPLE_CONTENT_TAG						= 0xFFFFFE1B;
	public static final String ELM_SIMPLE_TYPE_NAME						= "SimpleType";
	public static final int ELM_ROOT_TAG									= 0xFFFFFE01;

	public static final int ATTR_DICT_NUMBER_TAG							= 0xFFFFFE00;
	public static final String ATTR_DICT_NUMBER_TAG_NAME				= "DictNumber";
	public static final int ATTR_COLLECTION_NUMBER_TAG					= 0xFFFFFE01;
	public static final String ATTR_COLLECTION_NUMBER_TAG_NAME		= "CollectionNumber";
	public static final int ATTR_COLLECTION_NAME_TAG					= 0xFFFFFE02;
	public static final String ATTR_COLLECTION_NAME_TAG_NAME			= "CollectionName";
	public static final int ATTR_NAME_TAG									= 0xFFFFFE03;
	public static final String ATTR_NAME_TAG_NAME						= "name";
	public static final int ATTR_TARGET_NAMESPACE_TAG					= 0xFFFFFE04;
	public static final String ATTR_TARGET_NAMESPACE_TAG_NAME		= "targetNameSpace";
	public static final int ATTR_TYPE_TAG									= 0xFFFFFE05;
	public static final String ATTR_TYPE_TAG_NAME						= "type";
	public static final int ATTR_STATE_TAG									= 0xFFFFFE06;
	public static final String ATTR_STATE_TAG_NAME						= "State";
	public static final int ATTR_LANGUAGE_TAG								= 0xFFFFFE07;
	public static final String ATTR_LANGUAGE_TAG_NAME					= "Language";
	public static final int ATTR_INDEX_OPTIONS_TAG						= 0xFFFFFE08;
	public static final String ATTR_INDEX_OPTIONS_TAG_NAME			= "IndexOptions";
	public static final int ATTR_INDEX_ON_TAG								= 0xFFFFFE09;
	public static final String ATTR_INDEX_ON_TAG_NAME					= "IndexOn";
	public static final int ATTR_REQUIRED_TAG								= 0xFFFFFE0A;
	public static final String ATTR_REQUIRED_TAG_NAME					= "Required";
	public static final int ATTR_LIMIT_TAG									= 0xFFFFFE0B;
	public static final String ATTR_LIMIT_TAG_NAME						= "Limit";
	public static final int ATTR_COMPARE_RULES_TAG						= 0xFFFFFE0C;
	public static final String ATTR_COMPARE_RULES_TAG_NAME			= "CompareRules";
	public static final int ATTR_KEY_COMPONENT_TAG						= 0xFFFFFE0D;
	public static final String ATTR_KEY_COMPONENT_TAG_NAME			= "KeyComponent";
	public static final int ATTR_DATA_COMPONENT_TAG						= 0xFFFFFE0E;
	public static final String ATTR_DATA_COMPONENT_TAG_NAME			= "DataComponent";
	public static final int ATTR_LAST_DOC_INDEXED_TAG					= 0xFFFFFE0F;
	public static final String ATTR_LAST_DOC_INDEXED_TAG_NAME		= "LastDocumentIndexed";
	public static final int ATTR_NEXT_ELEMENT_NUM_TAG					= 0xFFFFFE10;
	public static final String ATTR_NEXT_ELEMENT_NUM_TAG_NAME		= "NextElementNum";
	public static final int ATTR_NEXT_ATTRIBUTE_NUM_TAG				= 0xFFFFFE11;
	public static final String ATTR_NEXT_ATTRIBUTE_NUM_TAG_NAME		= "NextAttributeNum";
	public static final int ATTR_NEXT_INDEX_NUM_TAG						= 0xFFFFFE12;
	public static final String ATTR_NEXT_INDEX_NUM_TAG_NAME			= "NextIndexNum";
	public static final int ATTR_NEXT_COLLECTION_NUM_TAG				= 0xFFFFFE13;
	public static final String ATTR_NEXT_COLLECTION_NUM_TAG_NAME	= "NextCollectionNum";
	public static final int ATTR_NEXT_PREFIX_NUM_TAG					= 0xFFFFFE14;
	public static final String ATTR_NEXT_PREFIX_NUM_TAG_NAME			= "NextPrefixNum";
	public static final int ATTR_DEFAULT_PREFIX_TAG						= 0xFFFFFE15;
	public static final String ATTR_DEFAULT_PREFIX_TAG_NAME			= "DefaultPrefix";
	public static final int ATTR_SOURCE_TAG								= 0xFFFFFE16;
	public static final String ATTR_SOURCE_TAG_NAME						= "Source";
	public static final int ATTR_STATE_CHANGE_COUNT_TAG				= 0xFFFFFE17;
	public static final String ATTR_STATE_CHANGE_COUNT_TAG_NAME		= "StateChangeCount";
	public static final int ATTR_XMLNS_TAG									= 0xFFFFFE18;
	public static final String ATTR_XMLNS_TAG_NAME						= "xmlns";
	public static final int ATTR_ABSTRACT_TAG								= 0xFFFFFE19;
	public static final String ATTR_ABSTRACT_TAG_NAME					= "abstract";
	public static final int ATTR_BASE_TAG									= 0xFFFFFE1A;
	public static final String ATTR_BASE_TAG_NAME						= "base";
	public static final int ATTR_BLOCK_TAG									= 0xFFFFFE1B;
	public static final String ATTR_BLOCK_TAG_NAME						= "block";
	public static final int ATTR_DEFAULT_TAG								= 0xFFFFFE1C;
	public static final String ATTR_DEFAULT_TAG_NAME					= "default";
	public static final int ATTR_FINAL_TAG									= 0xFFFFFE1D;
	public static final String ATTR_FINAL_TAG_NAME						= "final";
	public static final int ATTR_FIXED_TAG									= 0xFFFFFE1E;
	public static final String ATTR_FIXED_TAG_NAME						= "fixed";
	public static final int ATTR_ITEM_TYPE_TAG							= 0xFFFFFE1F;
	public static final String ATTR_ITEM_TYPE_TAG_NAME					= "itemtype";
	public static final int ATTR_MAX_OCCURS_TAG							= 0xFFFFFE20;
	public static final String ATTR_MAX_OCCURS_TAG_NAME				= "maxoccurs";
	public static final int ATTR_MEMBER_TYPES_TAG						= 0xFFFFFE21;
	public static final String ATTR_MEMBER_TYPES_TAG_NAME				= "membertypes";
	public static final int ATTR_MIN_OCCURS_TAG							= 0xFFFFFE22;
	public static final String ATTR_MIN_OCCURS_TAG_NAME				= "minoccurs";
	public static final int ATTR_MIXED_TAG									= 0xFFFFFE23;
	public static final String ATTR_MIXED_TAG_NAME						= "mixed";
	public static final int ATTR_NILLABLE_TAG								= 0xFFFFFE24;
	public static final String ATTR_NILLABLE_TAG_NAME					= "nillable";
	public static final int ATTR_REF_TAG									= 0xFFFFFE25;
	public static final String ATTR_REF_TAG_NAME							= "ref";
	public static final int ATTR_USE_TAG									= 0xFFFFFE26;
	public static final String ATTR_USE_TAG_NAME							= "use";
	public static final int ATTR_VALUE_TAG									= 0xFFFFFE27;
	public static final String ATTR_VALUE_TAG_NAME						= "value";

	public static final int XS_PREFIX_ID									= 0xFFFFFE00;
	public static final int XSI_PREFIX_ID									= 0xFFFFFE01;
	public static final int XFLAIM_PREFIX_ID								= 0xFFFFFE02;
	public static final int XMLNS_PREFIX_ID								= 0xFFFFFE03;
}
