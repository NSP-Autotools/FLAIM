//------------------------------------------------------------------------------
//	Desc:	XML parser
// Tabs:	3
//
// Copyright (c) 2000-2007 Novell, Inc. All Rights Reserved.
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

#include "ftksys.h"

#define FLM_XML_BASE_CHAR			0x01
#define FLM_XML_IDEOGRAPHIC		0x02
#define FLM_XML_COMBINING_CHAR	0x04
#define FLM_XML_DIGIT				0x08
#define FLM_XML_EXTENDER			0x10
#define FLM_XML_WHITESPACE			0x20

typedef struct
{
	char *			pszEntity;
	FLMUINT			uiValue;
} CharEntity;

typedef struct
{
	FLMUNICODE		uLowChar;
	FLMUNICODE		uHighChar;
	FLMUINT16		ui16Flag;
} CHAR_TBL;

static CHAR_TBL charTbl[] = 
{
	{ 0x0041, 0x005A, FLM_XML_BASE_CHAR},
	{ 0x0061, 0x007A, FLM_XML_BASE_CHAR},
	{ 0x00C0, 0x00D6, FLM_XML_BASE_CHAR},
	{ 0x00D8, 0x00F6, FLM_XML_BASE_CHAR},
	{ 0x00F8, 0x00FF, FLM_XML_BASE_CHAR},
	{ 0x0100, 0x0131, FLM_XML_BASE_CHAR},
	{ 0x0134, 0x013E, FLM_XML_BASE_CHAR},
	{ 0x0141, 0x0148, FLM_XML_BASE_CHAR},
	{ 0x014A, 0x017E, FLM_XML_BASE_CHAR},
	{ 0x0180, 0x01C3, FLM_XML_BASE_CHAR},
	{ 0x01CD, 0x01F0, FLM_XML_BASE_CHAR},
	{ 0x01F4, 0x01F5, FLM_XML_BASE_CHAR},
	{ 0x01FA, 0x0217, FLM_XML_BASE_CHAR},
	{ 0x0250, 0x02A8, FLM_XML_BASE_CHAR},
	{ 0x02BB, 0x02C1, FLM_XML_BASE_CHAR},
	{ 0x0386, 0x0386, FLM_XML_BASE_CHAR},
	{ 0x0388, 0x038A, FLM_XML_BASE_CHAR},
	{ 0x038C, 0x038C, FLM_XML_BASE_CHAR},
	{ 0x038E, 0x03A1, FLM_XML_BASE_CHAR},
	{ 0x03A3, 0x03CE, FLM_XML_BASE_CHAR},
	{ 0x03D0, 0x03D6, FLM_XML_BASE_CHAR},
	{ 0x03DA, 0x03DA, FLM_XML_BASE_CHAR},
	{ 0x03DC, 0x03DC, FLM_XML_BASE_CHAR},
	{ 0x03DE, 0x03DE, FLM_XML_BASE_CHAR},
	{ 0x03E0, 0x03E0, FLM_XML_BASE_CHAR},
	{ 0x03E2, 0x03F3, FLM_XML_BASE_CHAR},
	{ 0x0401, 0x040C, FLM_XML_BASE_CHAR},
	{ 0x040E, 0x044F, FLM_XML_BASE_CHAR},
	{ 0x0451, 0x045C, FLM_XML_BASE_CHAR},
	{ 0x045E, 0x0481, FLM_XML_BASE_CHAR},
	{ 0x0490, 0x04C4, FLM_XML_BASE_CHAR},
	{ 0x04C7, 0x04C8, FLM_XML_BASE_CHAR},
	{ 0x04CB, 0x04CC, FLM_XML_BASE_CHAR},
	{ 0x04D0, 0x04EB, FLM_XML_BASE_CHAR},
	{ 0x04EE, 0x04F5, FLM_XML_BASE_CHAR},
	{ 0x04F8, 0x04F9, FLM_XML_BASE_CHAR},
	{ 0x0531, 0x0556, FLM_XML_BASE_CHAR},
	{ 0x0559, 0x0559, FLM_XML_BASE_CHAR},
	{ 0x0561, 0x0586, FLM_XML_BASE_CHAR},
	{ 0x05D0, 0x05EA, FLM_XML_BASE_CHAR},
	{ 0x05F0, 0x05F2, FLM_XML_BASE_CHAR},
	{ 0x0621, 0x063A, FLM_XML_BASE_CHAR},
	{ 0x0641, 0x06B7, FLM_XML_BASE_CHAR},
	{ 0x06BA, 0x06BE, FLM_XML_BASE_CHAR},
	{ 0x06C0, 0x06CE, FLM_XML_BASE_CHAR},
	{ 0x06D0, 0x06D3, FLM_XML_BASE_CHAR},
	{ 0x06D5, 0x06D5, FLM_XML_BASE_CHAR},
	{ 0x06E5, 0x06E6, FLM_XML_BASE_CHAR},
	{ 0x0905, 0x0939, FLM_XML_BASE_CHAR},
	{ 0x093D, 0x093D, FLM_XML_BASE_CHAR},
	{ 0x0958, 0x0961, FLM_XML_BASE_CHAR},
	{ 0x0985, 0x098C, FLM_XML_BASE_CHAR},
	{ 0x098F, 0x0990, FLM_XML_BASE_CHAR},
	{ 0x0993, 0x09A8, FLM_XML_BASE_CHAR},
	{ 0x09AA, 0x09B0, FLM_XML_BASE_CHAR},
	{ 0x09B2, 0x09B2, FLM_XML_BASE_CHAR},
	{ 0x09B6, 0x09B9, FLM_XML_BASE_CHAR},
	{ 0x0061, 0x007A, FLM_XML_BASE_CHAR},
	{ 0x09DC, 0x09DD, FLM_XML_BASE_CHAR},
	{ 0x09DF, 0x09E1, FLM_XML_BASE_CHAR},
	{ 0x09F0, 0x09F1, FLM_XML_BASE_CHAR},
	{ 0x0A05, 0x0A0A, FLM_XML_BASE_CHAR},
	{ 0x0A0F, 0x0A10, FLM_XML_BASE_CHAR},
	{ 0x0A13, 0x0A28, FLM_XML_BASE_CHAR},
	{ 0x0A2A, 0x0A30, FLM_XML_BASE_CHAR},
	{ 0x0A32, 0x0A33, FLM_XML_BASE_CHAR},
	{ 0x0A35, 0x0A36, FLM_XML_BASE_CHAR},
	{ 0x0A38, 0x0A39, FLM_XML_BASE_CHAR},
	{ 0x0A59, 0x0A5C, FLM_XML_BASE_CHAR},
	{ 0x0A5E, 0x0A5E, FLM_XML_BASE_CHAR},
	{ 0x0A72, 0x0A74, FLM_XML_BASE_CHAR},
	{ 0x0A85, 0x0A8B, FLM_XML_BASE_CHAR},
	{ 0x0A8D, 0x0A8D, FLM_XML_BASE_CHAR},
	{ 0x0A8F, 0x0A91, FLM_XML_BASE_CHAR},
	{ 0x0A93, 0x0AA8, FLM_XML_BASE_CHAR},
	{ 0x0AAA, 0x0AB0, FLM_XML_BASE_CHAR},
	{ 0x0AB2, 0x0AB3, FLM_XML_BASE_CHAR},
	{ 0x0AB5, 0x0AB9, FLM_XML_BASE_CHAR},
	{ 0x0ABD, 0x0ABD, FLM_XML_BASE_CHAR},
	{ 0x0AE0, 0x0AE0, FLM_XML_BASE_CHAR},
	{ 0x0B05, 0x0B0C, FLM_XML_BASE_CHAR},
	{ 0x0B0F, 0x0B10, FLM_XML_BASE_CHAR},
	{ 0x0B13, 0x0B28, FLM_XML_BASE_CHAR},
	{ 0x0B2A, 0x0B30, FLM_XML_BASE_CHAR},
	{ 0x0B32, 0x0B33, FLM_XML_BASE_CHAR},
	{ 0x0B36, 0x0B39, FLM_XML_BASE_CHAR},
	{ 0x0B3D, 0x0B3D, FLM_XML_BASE_CHAR},
	{ 0x0B5C, 0x0B5D, FLM_XML_BASE_CHAR},
	{ 0x0B5F, 0x0B61, FLM_XML_BASE_CHAR},
	{ 0x0B85, 0x0B8A, FLM_XML_BASE_CHAR},
	{ 0x0B8E, 0x0B90, FLM_XML_BASE_CHAR},
	{ 0x0B92, 0x0B95, FLM_XML_BASE_CHAR},
	{ 0x0B99, 0x0B9A, FLM_XML_BASE_CHAR},
	{ 0x0B9C, 0x0B9C, FLM_XML_BASE_CHAR},
	{ 0x0B9E, 0x0B9F, FLM_XML_BASE_CHAR},
	{ 0x0BA3, 0x0BA4, FLM_XML_BASE_CHAR},
	{ 0x0BA8, 0x0BAA, FLM_XML_BASE_CHAR},
	{ 0x0BAE, 0x0BB5, FLM_XML_BASE_CHAR},
	{ 0x0BB7, 0x0BB9, FLM_XML_BASE_CHAR},
	{ 0x0C05, 0x0C0C, FLM_XML_BASE_CHAR},
	{ 0x0C0E, 0x0C10, FLM_XML_BASE_CHAR},
	{ 0x0C12, 0x0C28, FLM_XML_BASE_CHAR},
	{ 0x0C2A, 0x0C33, FLM_XML_BASE_CHAR},
	{ 0x0C35, 0x0C39, FLM_XML_BASE_CHAR},
	{ 0x0C60, 0x0C61, FLM_XML_BASE_CHAR},
	{ 0x0C85, 0x0C8C, FLM_XML_BASE_CHAR},
	{ 0x0C8E, 0x0C90, FLM_XML_BASE_CHAR},
	{ 0x0C92, 0x0CA8, FLM_XML_BASE_CHAR},
	{ 0x0CAA, 0x0CB3, FLM_XML_BASE_CHAR},
	{ 0x0CB5, 0x0CB9, FLM_XML_BASE_CHAR},
	{ 0x0CDE, 0x0CDE, FLM_XML_BASE_CHAR},
	{ 0x0CE0, 0x0CE1, FLM_XML_BASE_CHAR},
	{ 0x0D05, 0x0D0C, FLM_XML_BASE_CHAR},
	{ 0x0D0E, 0x0D10, FLM_XML_BASE_CHAR},
	{ 0x0D12, 0x0D28, FLM_XML_BASE_CHAR},
	{ 0x0D2A, 0x0D39, FLM_XML_BASE_CHAR},
	{ 0x0D60, 0x0D61, FLM_XML_BASE_CHAR},
	{ 0x0E01, 0x0E2E, FLM_XML_BASE_CHAR},
	{ 0x0E30, 0x0E30, FLM_XML_BASE_CHAR},
	{ 0x0E32, 0x0E33, FLM_XML_BASE_CHAR},
	{ 0x0E40, 0x0E45, FLM_XML_BASE_CHAR},
	{ 0x0E81, 0x0E82, FLM_XML_BASE_CHAR},
	{ 0x0E84, 0x0E84, FLM_XML_BASE_CHAR},
	{ 0x0E87, 0x0E88, FLM_XML_BASE_CHAR},
	{ 0x0E8A, 0x0E8A, FLM_XML_BASE_CHAR},
	{ 0x0E8D, 0x0E8D, FLM_XML_BASE_CHAR},
	{ 0x0E94, 0x0E97, FLM_XML_BASE_CHAR},
	{ 0x0E99, 0x0E9F, FLM_XML_BASE_CHAR},
	{ 0x0EA1, 0x0EA3, FLM_XML_BASE_CHAR},
	{ 0x0EA5, 0x0EA5, FLM_XML_BASE_CHAR},
	{ 0x0EA7, 0x0EA7, FLM_XML_BASE_CHAR},
	{ 0x0EAA, 0x0EAB, FLM_XML_BASE_CHAR},
	{ 0x0EAD, 0x0EAE, FLM_XML_BASE_CHAR},
	{ 0x0EB0, 0x0EB0, FLM_XML_BASE_CHAR},
	{ 0x0EB2, 0x0EB3, FLM_XML_BASE_CHAR},
	{ 0x0EBD, 0x0EBD, FLM_XML_BASE_CHAR},
	{ 0x0EC0, 0x0EC4, FLM_XML_BASE_CHAR},
	{ 0x0F40, 0x0F47, FLM_XML_BASE_CHAR},
	{ 0x0F49, 0x0F69, FLM_XML_BASE_CHAR},
	{ 0x10A0, 0x10C5, FLM_XML_BASE_CHAR},
	{ 0x10D0, 0x10F6, FLM_XML_BASE_CHAR},
	{ 0x1100, 0x1100, FLM_XML_BASE_CHAR},
	{ 0x1102, 0x1103, FLM_XML_BASE_CHAR},
	{ 0x1105, 0x1107, FLM_XML_BASE_CHAR},
	{ 0x1109, 0x1109, FLM_XML_BASE_CHAR},
	{ 0x110B, 0x110C, FLM_XML_BASE_CHAR},
	{ 0x110E, 0x1112, FLM_XML_BASE_CHAR},
	{ 0x113C, 0x113C, FLM_XML_BASE_CHAR},
	{ 0x113E, 0x113E, FLM_XML_BASE_CHAR},
	{ 0x1140, 0x1140, FLM_XML_BASE_CHAR},
	{ 0x114C, 0x114C, FLM_XML_BASE_CHAR},
	{ 0x114E, 0x114E, FLM_XML_BASE_CHAR},
	{ 0x1150, 0x1150, FLM_XML_BASE_CHAR},
	{ 0x1154, 0x1155, FLM_XML_BASE_CHAR},
	{ 0x1159, 0x1159, FLM_XML_BASE_CHAR},
	{ 0x115F, 0x1161, FLM_XML_BASE_CHAR},
	{ 0x1163, 0x1163, FLM_XML_BASE_CHAR},
	{ 0x1165, 0x1165, FLM_XML_BASE_CHAR},
	{ 0x1167, 0x1167, FLM_XML_BASE_CHAR},
	{ 0x1169, 0x1169, FLM_XML_BASE_CHAR},
	{ 0x116D, 0x116E, FLM_XML_BASE_CHAR},
	{ 0x1172, 0x1173, FLM_XML_BASE_CHAR},
	{ 0x1175, 0x1175, FLM_XML_BASE_CHAR},
	{ 0x119E, 0x119E, FLM_XML_BASE_CHAR},
	{ 0x11A8, 0x11A8, FLM_XML_BASE_CHAR},
	{ 0x11AB, 0x11AB, FLM_XML_BASE_CHAR},
	{ 0x11AE, 0x11AF, FLM_XML_BASE_CHAR},
	{ 0x11B7, 0x11B8, FLM_XML_BASE_CHAR},
	{ 0x11BA, 0x11BA, FLM_XML_BASE_CHAR},
	{ 0x11BC, 0x11C2, FLM_XML_BASE_CHAR},
	{ 0x11EB, 0x11EB, FLM_XML_BASE_CHAR},
	{ 0x11F0, 0x11F0, FLM_XML_BASE_CHAR},
	{ 0x11F9, 0x11F9, FLM_XML_BASE_CHAR},
	{ 0x1E00, 0x1E9B, FLM_XML_BASE_CHAR},
	{ 0x1EA0, 0x1EF9, FLM_XML_BASE_CHAR},
	{ 0x1F00, 0x1F15, FLM_XML_BASE_CHAR},
	{ 0x1F18, 0x1F1D, FLM_XML_BASE_CHAR},
	{ 0x1F20, 0x1F45, FLM_XML_BASE_CHAR},
	{ 0x1F48, 0x1F4D, FLM_XML_BASE_CHAR},
	{ 0x1F50, 0x1F57, FLM_XML_BASE_CHAR},
	{ 0x1F59, 0x1F59, FLM_XML_BASE_CHAR},
	{ 0x1F5B, 0x1F5B, FLM_XML_BASE_CHAR},
	{ 0x1F5D, 0x1F5D, FLM_XML_BASE_CHAR},
	{ 0x1F5F, 0x1F7D, FLM_XML_BASE_CHAR},
	{ 0x1F80, 0x1FB4, FLM_XML_BASE_CHAR},
	{ 0x1FB6, 0x1FBC, FLM_XML_BASE_CHAR},
	{ 0x1FBE, 0x1FBE, FLM_XML_BASE_CHAR},
	{ 0x1FC2, 0x1FC4, FLM_XML_BASE_CHAR},
	{ 0x1FC6, 0x1FCC, FLM_XML_BASE_CHAR},
	{ 0x1FD0, 0x1FD3, FLM_XML_BASE_CHAR},
	{ 0x1FD6, 0x1FDB, FLM_XML_BASE_CHAR},
	{ 0x1FE0, 0x1FEC, FLM_XML_BASE_CHAR},
	{ 0x1FF2, 0x1FF4, FLM_XML_BASE_CHAR},
	{ 0x1FF6, 0x1FFC, FLM_XML_BASE_CHAR},
	{ 0x2126, 0x2126, FLM_XML_BASE_CHAR},
	{ 0x212A, 0x212B, FLM_XML_BASE_CHAR},
	{ 0x212E, 0x212E, FLM_XML_BASE_CHAR},
	{ 0x2180, 0x2182, FLM_XML_BASE_CHAR},
	{ 0x3041, 0x3094, FLM_XML_BASE_CHAR},
	{ 0x30A1, 0x30FA, FLM_XML_BASE_CHAR},
	{ 0x3105, 0x312C, FLM_XML_BASE_CHAR},
	{ 0xAC00, 0xD7A3, FLM_XML_BASE_CHAR},

	{ 0x4E00, 0x9FA5, FLM_XML_IDEOGRAPHIC},
	{ 0x3007, 0x3007, FLM_XML_IDEOGRAPHIC},
	{ 0x3021, 0x3029, FLM_XML_IDEOGRAPHIC},

	{ 0x0300, 0x0345, FLM_XML_COMBINING_CHAR},
	{ 0x0360, 0x0361, FLM_XML_COMBINING_CHAR},
	{ 0x0483, 0x0486, FLM_XML_COMBINING_CHAR},
	{ 0x0591, 0x05A1, FLM_XML_COMBINING_CHAR},
	{ 0x05A3, 0x05B9, FLM_XML_COMBINING_CHAR},
	{ 0x05BB, 0x05BD, FLM_XML_COMBINING_CHAR},
	{ 0x05BF, 0x05BF, FLM_XML_COMBINING_CHAR},
	{ 0x05C1, 0x05C2, FLM_XML_COMBINING_CHAR},
	{ 0x05C4, 0x05C4, FLM_XML_COMBINING_CHAR},
	{ 0x064B, 0x0652, FLM_XML_COMBINING_CHAR},
	{ 0x0670, 0x0670, FLM_XML_COMBINING_CHAR},
	{ 0x06D6, 0x06DC, FLM_XML_COMBINING_CHAR},
	{ 0x06DD, 0x06DF, FLM_XML_COMBINING_CHAR},
	{ 0x06E0, 0x06E4, FLM_XML_COMBINING_CHAR},
	{ 0x06E7, 0x06E8, FLM_XML_COMBINING_CHAR},
	{ 0x06EA, 0x06ED, FLM_XML_COMBINING_CHAR},
	{ 0x0901, 0x0903, FLM_XML_COMBINING_CHAR},
	{ 0x093C, 0x093C, FLM_XML_COMBINING_CHAR},
	{ 0x093E, 0x094C, FLM_XML_COMBINING_CHAR},
	{ 0x094D, 0x094D, FLM_XML_COMBINING_CHAR},
	{ 0x0951, 0x0954, FLM_XML_COMBINING_CHAR},
	{ 0x0962, 0x0963, FLM_XML_COMBINING_CHAR},
	{ 0x0981, 0x0983, FLM_XML_COMBINING_CHAR},
	{ 0x09BC, 0x09BC, FLM_XML_COMBINING_CHAR},
	{ 0x09BE, 0x09BE, FLM_XML_COMBINING_CHAR},
	{ 0x09BF, 0x09BF, FLM_XML_COMBINING_CHAR},
	{ 0x09C0, 0x09C4, FLM_XML_COMBINING_CHAR},
	{ 0x09C7, 0x09C8, FLM_XML_COMBINING_CHAR},
	{ 0x09CB, 0x09CD, FLM_XML_COMBINING_CHAR},
	{ 0x09D7, 0x09D7, FLM_XML_COMBINING_CHAR},
	{ 0x09E2, 0x09E3, FLM_XML_COMBINING_CHAR},
	{ 0x0A02, 0x0A02, FLM_XML_COMBINING_CHAR},
	{ 0x0A3C, 0x0A3C, FLM_XML_COMBINING_CHAR},
	{ 0x0A3E, 0x0A3E, FLM_XML_COMBINING_CHAR},
	{ 0x0A3F, 0x0A3F, FLM_XML_COMBINING_CHAR},
	{ 0x0A40, 0x0A42, FLM_XML_COMBINING_CHAR},
	{ 0x0A47, 0x0A48, FLM_XML_COMBINING_CHAR},
	{ 0x0A4B, 0x0A4D, FLM_XML_COMBINING_CHAR},
	{ 0x0A70, 0x0A71, FLM_XML_COMBINING_CHAR},
	{ 0x0A81, 0x0A83, FLM_XML_COMBINING_CHAR},
	{ 0x0ABC, 0x0ABC, FLM_XML_COMBINING_CHAR},
	{ 0x0ABE, 0x0AC5, FLM_XML_COMBINING_CHAR},
	{ 0x0AC7, 0x0AC9, FLM_XML_COMBINING_CHAR},
	{ 0x0ACB, 0x0ACD, FLM_XML_COMBINING_CHAR},
	{ 0x0B01, 0x0B03, FLM_XML_COMBINING_CHAR},
	{ 0x0B3C, 0x0B3C, FLM_XML_COMBINING_CHAR},
	{ 0x0B3E, 0x0B43, FLM_XML_COMBINING_CHAR},
	{ 0x0B47, 0x0B48, FLM_XML_COMBINING_CHAR},
	{ 0x0B4B, 0x0B4D, FLM_XML_COMBINING_CHAR},
	{ 0x0B56, 0x0B57, FLM_XML_COMBINING_CHAR},
	{ 0x0B82, 0x0B83, FLM_XML_COMBINING_CHAR},
	{ 0x0BBE, 0x0BC2, FLM_XML_COMBINING_CHAR},
	{ 0x0BC6, 0x0BC8, FLM_XML_COMBINING_CHAR},
	{ 0x0BCA, 0x0BCD, FLM_XML_COMBINING_CHAR},
	{ 0x0BD7, 0x0BD7, FLM_XML_COMBINING_CHAR},
	{ 0x0C01, 0x0C03, FLM_XML_COMBINING_CHAR},
	{ 0x0C3E, 0x0C44, FLM_XML_COMBINING_CHAR},
	{ 0x0C46, 0x0C48, FLM_XML_COMBINING_CHAR},
	{ 0x0C4A, 0x0C4D, FLM_XML_COMBINING_CHAR},
	{ 0x0C55, 0x0C56, FLM_XML_COMBINING_CHAR},
	{ 0x0C82, 0x0C83, FLM_XML_COMBINING_CHAR},
	{ 0x0CBE, 0x0CC4, FLM_XML_COMBINING_CHAR},
	{ 0x0CC6, 0x0CC8, FLM_XML_COMBINING_CHAR},
	{ 0x0CCA, 0x0CCD, FLM_XML_COMBINING_CHAR},
	{ 0x0CD5, 0x0CD6, FLM_XML_COMBINING_CHAR},
	{ 0x0D02, 0x0D03, FLM_XML_COMBINING_CHAR},
	{ 0x0D3E, 0x0D43, FLM_XML_COMBINING_CHAR},
	{ 0x0D46, 0x0D48, FLM_XML_COMBINING_CHAR},
	{ 0x0D4A, 0x0D4D, FLM_XML_COMBINING_CHAR},
	{ 0x0D57, 0x0D57, FLM_XML_COMBINING_CHAR},
	{ 0x0E31, 0x0E31, FLM_XML_COMBINING_CHAR},
	{ 0x0E34, 0x0E3A, FLM_XML_COMBINING_CHAR},
	{ 0x0E47, 0x0E4E, FLM_XML_COMBINING_CHAR},
	{ 0x0EB1, 0x0EB1, FLM_XML_COMBINING_CHAR},
	{ 0x0EB4, 0x0EB9, FLM_XML_COMBINING_CHAR},
	{ 0x0EBB, 0x0EBC, FLM_XML_COMBINING_CHAR},
	{ 0x0EC8, 0x0ECD, FLM_XML_COMBINING_CHAR},
	{ 0x0F18, 0x0F19, FLM_XML_COMBINING_CHAR},
	{ 0x0F35, 0x0F35, FLM_XML_COMBINING_CHAR},
	{ 0x0F37, 0x0F37, FLM_XML_COMBINING_CHAR},
	{ 0x0F39, 0x0F39, FLM_XML_COMBINING_CHAR},
	{ 0x0F3E, 0x0F3E, FLM_XML_COMBINING_CHAR},
	{ 0x0F3F, 0x0F3F, FLM_XML_COMBINING_CHAR},
	{ 0x0F71, 0x0F84, FLM_XML_COMBINING_CHAR},
	{ 0x0F86, 0x0F8B, FLM_XML_COMBINING_CHAR},
	{ 0x0F90, 0x0F95, FLM_XML_COMBINING_CHAR},
	{ 0x0F97, 0x0F97, FLM_XML_COMBINING_CHAR},
	{ 0x0F99, 0x0FAD, FLM_XML_COMBINING_CHAR},
	{ 0x0FB1, 0x0FB7, FLM_XML_COMBINING_CHAR},
	{ 0x0FB9, 0x0FB9, FLM_XML_COMBINING_CHAR},
	{ 0x20D0, 0x20DC, FLM_XML_COMBINING_CHAR},
	{ 0x20E1, 0x20E1, FLM_XML_COMBINING_CHAR},
	{ 0x302A, 0x302F, FLM_XML_COMBINING_CHAR},
	{ 0x3099, 0x3099, FLM_XML_COMBINING_CHAR},
	{ 0x309A, 0x309A, FLM_XML_COMBINING_CHAR},

	{ 0x0030, 0x0039, FLM_XML_DIGIT},
	{ 0x0660, 0x0669, FLM_XML_DIGIT},
	{ 0x06F0, 0x06F9, FLM_XML_DIGIT},
	{ 0x0966, 0x096F, FLM_XML_DIGIT},
	{ 0x09E6, 0x09EF, FLM_XML_DIGIT},
	{ 0x0A66, 0x0A6F, FLM_XML_DIGIT},
	{ 0x0AE6, 0x0AEF, FLM_XML_DIGIT},
	{ 0x0B66, 0x0B6F, FLM_XML_DIGIT},
	{ 0x0BE7, 0x0BEF, FLM_XML_DIGIT},
	{ 0x0C66, 0x0C6F, FLM_XML_DIGIT},
	{ 0x0CE6, 0x0CEF, FLM_XML_DIGIT},
	{ 0x0D66, 0x0D6F, FLM_XML_DIGIT},
	{ 0x0E50, 0x0E59, FLM_XML_DIGIT},
	{ 0x0ED0, 0x0ED9, FLM_XML_DIGIT},
	{ 0x0F20, 0x0F29, FLM_XML_DIGIT},

	{ 0x00B7, 0x00B7, FLM_XML_EXTENDER},
	{ 0x02D0, 0x02D0, FLM_XML_EXTENDER},
	{ 0x02D1, 0x02D1, FLM_XML_EXTENDER},
	{ 0x0387, 0x0387, FLM_XML_EXTENDER},
	{ 0x0640, 0x0640, FLM_XML_EXTENDER},
	{ 0x0E46, 0x0E46, FLM_XML_EXTENDER},
	{ 0x0EC6, 0x0EC6, FLM_XML_EXTENDER},
	{ 0x3005, 0x3005, FLM_XML_EXTENDER},
	{ 0x3031, 0x3035, FLM_XML_EXTENDER},
	{ 0x309D, 0x309E, FLM_XML_EXTENDER},
	{ 0x30FC, 0x30FE, FLM_XML_EXTENDER},

	{ 0x0009, 0x0009, FLM_XML_WHITESPACE},
	{ 0x000A, 0x000A, FLM_XML_WHITESPACE},
	{ 0x000D, 0x000D, FLM_XML_WHITESPACE},
	{ 0x0020, 0x0020, FLM_XML_WHITESPACE},
   { 0, 0, 0}
};

/****************************************************************************
Desc:
****************************************************************************/
F_XML::F_XML()
{
	m_pCharTable = NULL;
}

/****************************************************************************
Desc:
****************************************************************************/
F_XML::~F_XML()
{
	if( m_pCharTable)
	{
		f_free( &m_pCharTable);
	}
}

/****************************************************************************
Desc: Sets a character's type flag in the character lookup table
****************************************************************************/
void F_XML::setCharFlag(
	FLMUNICODE		uLowChar,
	FLMUNICODE		uHighChar,
	FLMUINT16		ui16Flag)
{
	FLMUINT		uiLoop;

	f_assert( uLowChar <= uHighChar);

	for( uiLoop = (FLMUINT)uLowChar; uiLoop <= (FLMUINT)uHighChar; uiLoop++)
	{
		m_pCharTable[ uiLoop].ucFlags |= (FLMBYTE)ui16Flag;
	}
}

/****************************************************************************
Desc: Builds a character lookup table
****************************************************************************/
RCODE FTKAPI F_XML::setup( void)
{
	RCODE			rc = NE_FLM_OK;
	FLMUINT		uiLoop;

	if( m_pCharTable)
	{
		f_free( &m_pCharTable);
	}

	if( RC_BAD( rc = f_calloc( sizeof( XMLCHAR) * 0xFFFF, &m_pCharTable)))
	{
		goto Exit;
	}

   for( uiLoop = 0; charTbl[uiLoop].ui16Flag; uiLoop++)
   {
      setCharFlag( charTbl[uiLoop].uLowChar,
                charTbl[uiLoop].uHighChar,
                charTbl[uiLoop].ui16Flag);
   }

Exit:

	return( rc);
}

/****************************************************************************
Desc: Returns TRUE if the character is a valid XML PubID character
****************************************************************************/
FLMBOOL FTKAPI F_XML::isPubidChar(
	FLMUNICODE		uChar)
{
	if( uChar == FLM_UNICODE_SPACE ||
		uChar == FLM_UNICODE_LINEFEED ||
		(uChar >= FLM_UNICODE_a && uChar <= FLM_UNICODE_z) ||
		(uChar >= FLM_UNICODE_A && uChar <= FLM_UNICODE_Z) ||
		(uChar >= FLM_UNICODE_0 && uChar <= FLM_UNICODE_9) ||
		uChar == FLM_UNICODE_HYPHEN ||
		uChar == FLM_UNICODE_APOS ||
		uChar == FLM_UNICODE_LPAREN ||
		uChar == FLM_UNICODE_RPAREN ||
		uChar == FLM_UNICODE_PLUS ||
		uChar == FLM_UNICODE_COMMA ||
		uChar == FLM_UNICODE_PERIOD ||
		uChar == FLM_UNICODE_FSLASH ||
		uChar == FLM_UNICODE_COLON ||
		uChar == FLM_UNICODE_EQ ||
		uChar == FLM_UNICODE_QUEST ||
		uChar == FLM_UNICODE_SEMI ||
		uChar == FLM_UNICODE_BANG ||
		uChar == FLM_UNICODE_ASTERISK ||
		uChar == FLM_UNICODE_POUND ||
		uChar == FLM_UNICODE_ATSIGN ||
		uChar == FLM_UNICODE_DOLLAR ||
		uChar == FLM_UNICODE_UNDERSCORE ||
		uChar == FLM_UNICODE_PERCENT)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a single or double quote character
****************************************************************************/
FLMBOOL FTKAPI F_XML::isQuoteChar(
	FLMUNICODE		uChar)
{
	if( uChar == FLM_UNICODE_QUOTE || uChar == FLM_UNICODE_APOS)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a whitespace character
****************************************************************************/
FLMBOOL FTKAPI F_XML::isWhitespace(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_WHITESPACE) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is an extender character
****************************************************************************/
FLMBOOL FTKAPI F_XML::isExtender(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_EXTENDER) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a combining character
****************************************************************************/
FLMBOOL FTKAPI F_XML::isCombiningChar(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_COMBINING_CHAR) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a valid XML naming character
****************************************************************************/
FLMBOOL FTKAPI F_XML::isNCNameChar(
	FLMUNICODE		uChar)
{
	if( isLetter( uChar) ||
		isDigit( uChar) ||
		uChar == FLM_UNICODE_PERIOD ||
		uChar == FLM_UNICODE_HYPHEN ||
		uChar == FLM_UNICODE_UNDERSCORE ||
		isCombiningChar( uChar) || isExtender( uChar))
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a valid XML naming character
****************************************************************************/
FLMBOOL FTKAPI F_XML::isNameChar(
	FLMUNICODE		uChar)
{
	if( isNCNameChar( uChar) ||
		uChar == FLM_UNICODE_COLON)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is an ideographic character
****************************************************************************/
FLMBOOL FTKAPI F_XML::isIdeographic(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_IDEOGRAPHIC) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a base character
****************************************************************************/
FLMBOOL FTKAPI F_XML::isBaseChar(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_BASE_CHAR) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a digit
****************************************************************************/
FLMBOOL FTKAPI F_XML::isDigit(
	FLMUNICODE		uChar)
{
	if( (m_pCharTable[ uChar].ucFlags & FLM_XML_DIGIT) != 0)
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc:		Returns TRUE if the character is a letter
****************************************************************************/
FLMBOOL FTKAPI F_XML::isLetter(
	FLMUNICODE		uChar)
{
	if( isBaseChar( uChar) || isIdeographic( uChar))
	{
		return( TRUE);
	}

	return( FALSE);
}

/****************************************************************************
Desc: 	Returns TRUE if the name is a valid XML name
****************************************************************************/
FLMBOOL FTKAPI F_XML::isNameValid(
	FLMUNICODE *	puzName,
	FLMBYTE *		pszName)
{
	FLMBOOL			bValid = FALSE;

	if( puzName)
	{
		FLMUNICODE *	puzTmp;

		if( !isLetter( *puzName) && *puzName != FLM_UNICODE_UNDERSCORE &&
			*puzName != FLM_UNICODE_COLON)
		{
			goto Exit;
		}

		puzTmp = &puzName[ 1];
		while( *puzTmp)
		{
			if( !isNameChar( *puzTmp))
			{
				goto Exit;
			}
			puzTmp++;
		}
	}

	if( pszName)
	{
		FLMBYTE *	pszTmp;

		if( !isLetter( *pszName) && *pszName != FLM_UNICODE_UNDERSCORE &&
			*pszName != FLM_UNICODE_COLON)
		{
			goto Exit;
		}

		pszTmp = &pszName[ 1];
		while( *pszTmp)
		{
			if( !isNameChar( *pszTmp))
			{
				goto Exit;
			}
			pszTmp++;
		}
	}

	bValid = TRUE;

Exit:

	return( bValid);
}
