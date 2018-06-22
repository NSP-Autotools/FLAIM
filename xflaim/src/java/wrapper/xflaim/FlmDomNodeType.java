//------------------------------------------------------------------------------
// Desc:	DOM Node Type
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
 * Provides enums for all of the possible DOM node types.
 * NOTE: The values in this class must match *exactly* with the
 * eFlmDomNodeType enum defined in xflaim.h
 */
public final class FlmDomNodeType 
{
	public final static int INVALID_NODE						=	0x00;
	public final static int DOCUMENT_NODE						=	0x01;
	public final static int ELEMENT_NODE						=	0x02;
	public final static int DATA_NODE							=	0x03;
	public final static int COMMENT_NODE						=	0x04;
	public final static int CDATA_SECTION_NODE				=	0x05;
	public final static int ANNOTATION_NODE					=	0x06;
	public final static int PROCESSING_INSTRUCTION_NODE	=	0x07;
	public final static int ATTRIBUTE_NODE						=	0x08;
	public final static int ANY_NODE_TYPE						=	0xFFFF;
}
