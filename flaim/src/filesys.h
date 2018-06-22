//-------------------------------------------------------------------------
// Desc: Definitions for internal database structure.
// Tabs: 3
//
// Copyright (c) 1990-1993, 1995-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FILESYS_H
#define FILESYS_H

#include "flaimsys.h"

#include "fpackon.h"

struct UCUR;

// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

	//		B-tree ELEMENT Definitions
	//
	//		Portable element definitions are the basic unit of storage within
	//		any b-tree block.	 The new format has changed to support longer
	//		keys up to 1023 bytes long.
	//
	//		KEY:
	//				BBE	- B-tree Block Element
	//				BNE	- B-tree Non-leaf element
	//				BUE	- B-tree Unknown Element - don't know if leaf or non-leaf
	//
	//				CONT	- the element 'does' continue to the next element
	//				DOMAIN- for the new reference set organization in non-leaf blks
	//				PKC	- number of bytes used in the previous key (PKC)
	//				KL		- key Length - # of bytes to represent the right most key info
	//				RL		- number of bytes in the record portion of the element
	//				KEY	- starts the key bounded by KEY_LEN
	//				CHILD_BLK - Three byte address of the childs block (non-leaf elms)
	//

	// BYTE 0 - BITS 0,1 - First and last element markers

	#define BBE_FIRST_FLAG		  0x80
	#define BBE_LAST_FLAG		  0x40

	#define BBE_IS_FIRST(elm)		((*(elm)) & BBE_FIRST_FLAG )
	#define BBE_NOT_FIRST(elm)		(!((*(elm)) & BBE_FIRST_FLAG ))
	#define BBE_SET_FIRST(elm)		((*(elm)) |= BBE_FIRST_FLAG)
	#define BBE_CLR_FIRST(elm)		((*(elm)) = (FLMBYTE)(*(elm) & ~(BBE_FIRST_FLAG)))

	#define BBE_IS_LAST(elm)		((*(elm)) & BBE_LAST_FLAG )
	#define BBE_NOT_LAST(elm)		(!((*(elm)) & BBE_LAST_FLAG ))
	#define BBE_SET_LAST(elm)		((*(elm)) |= (FLMBYTE)(BBE_LAST_FLAG))
	#define BBE_CLR_LAST(elm)		((*(elm)) = (FLMBYTE)(*(elm) & ~(BBE_LAST_FLAG)))

	#define BBE_IS_FIRST_LAST(e)	((*(e)) & (BBE_FIRST_FLAG|BBE_LAST_FLAG))

	#define BBE_MIDDLE_FLAG			(BBE_FIRST_FLAG|BBE_LAST_FLAG)
	#define BBE_IS_MIDDLE(elm)		(!((*(elm)) & (BBE_MIDDLE_FLAG)))
	#define BBE_SET_MIDDLE(elm)	((*(elm)) = (FLMBYTE)(*(elm) & ~(BBE_MIDDLE_FLAG)))

	// BYTE 0 - BITS 2,3 - Key Length High Bits

	#define BBE_KL_HBITS				0x30

	// BYTE 0 - BITS 4,5,6,7 - Previous Key Count - [0..15]

	#define BBE_PKC					0
	#define BBE_PKC_MAX				0x0F

	// BBE_SET_PKC should clear out all other values

	#define BBE_SET_PKC(elm,val)	((*(elm)) = (FLMBYTE)(BBE_PKC_MAX & (val)))
	#define BBE_CHK_PKC(val)		(((val) <= BBE_PKC_MAX) ? val : BBE_PKC_MAX)
	#define BBE_GET_PKC(elm)		((*(elm)) & BBE_PKC_MAX)
	#define BBE_GETR_PKC(elm)		(*(elm) & 0x3F)

	// BYTE 1 - Key Length

	#define BBE_KL						1
	#define BBE_SETR_KL(elm,val)	((elm)[BBE_KL] = (val))
	#define BBE_KL_SHIFT_BITS		4

	#define BBE_SET_KL(elm,val) \
	{ \
		if( (val) > 0xFF) \
			*(elm) |= (FLMBYTE)(((val) >> BBE_KL_SHIFT_BITS) & BBE_KL_HBITS); \
		(elm)[BBE_KL] = (FLMBYTE) (val); \
	}

	#define BBE_GETR_KL(elm)		((elm)[BBE_KL])
	#define BBE_GET_KL(elm)			(((*(elm) & BBE_KL_HBITS) << BBE_KL_SHIFT_BITS) + \
												(elm)[BBE_KL])

	// BYTE 2 - Record Length

	#define BBE_RL						2
	#define BBE_SET_RL(elm,val)	(((elm)[BBE_RL]) = (FLMBYTE)(val))
	#define BBE_GET_RL(elm)			((elm)[BBE_RL])

	// BYTE 3 - KEY

	#define BBE_KEY					3

	//
	// Non-leaf element format
	//

	// BYTE 0 - BIT 0 - DOMAIN FLAG

	#define BNE_DOMAIN				0x80
	#define BNE_IS_DOMAIN(elm)		((*(elm)) & BNE_DOMAIN)
	#define BNE_SET_DOMAIN(elm)	((*(elm)) |= BNE_DOMAIN)
	#define BNE_CLR_DOMAIN(elm)	((*(elm)) = *(elm) & (~(BNE_DOMAIN)))
	#define BNE_DOMAIN_LEN			3

	// BYTE 0 - BITS 1,2
	// Use BBE_KL_HBITS codes

	// BYTE 0 - Bits 3,4,5,6,7
	// Use BBE_xxx_PKC macros

	// BYTE 1
	// Use BBE_xxx_KL macros

	// BYTES 2-5 - CHILD BLOCK ADDRESS - 4 byte number

	#define BNE_CHILD_BLOCK			2
	#define BNE_CHILD_COUNT			6

	// BYTE 6 or 10 - Start of Key
	
	#define BNE_KEY_START			6
	#define BNE_KEY_COUNTS_START	10

	#define BNE_DATA_CHILD_BLOCK	4
	#define BNE_DATA_OVHD			8

	// The domain value in 3-byte high-low format will follow the key

	// GENERAL MANIPULATION MACROS
	//
	//		LEN	  - Length of element
	//		REC_OFS - Offset into the record portion (skip the key)
	//		REC_PTR - Address of where record portion starts
	//		KEY_OFS - Offset into where the key starts

	//
	// Compute the complete length of a leaf and non-leaf element
	//

	#define BBE_LEN(elm) \
		(BBE_GET_RL(elm) + BBE_GET_KL(elm) + BBE_KEY)

	#define BNE_LEN(stack,elm) \
		(BBE_GET_KL(elm) + stack->uiElmOvhd + \
		(BNE_IS_DOMAIN(elm) ? BNE_DOMAIN_LEN : 0))

	#define BBE_REC_OFS(elm) \
		(BBE_GET_KL(elm) + BBE_KEY)

	#define BBE_REC_PTR(elm) \
		(&(elm)[ BBE_REC_OFS(elm) ] )

// Record OPCODEs used in the storage of data record field values.
// All opcodes are prefixed by FOP which is Field OPcode.
//		l		 = bits used to represent the storage length of the field value
//		a		 = bit flag for number of bytes for TAG_NUM
//		b		 = bit flag for number of bytes for value length
//		ffff	 = 4 bits are used for the field type (0..15)
//		vvv	 = value of levels to shift out to (0..7)
//		c		 = The context flag - 0 is sibling - 1 is child
//		z		 = bits used for a compressed tNum (field number)
//		VALUE	 = value portion of the field
//		x		 = future flag
//		i		 = ID is 2 bytes (0) or 4 bytes long
//		e		 = Value is encrypted

	#define FOP_STANDARD						0			// Use 1 left bit	 0cll llll
																// zzzz zzzz
																// VALUE

	#define FOP_IS_STANDARD(p)				(!(*(p) & 0x80))
	#define FSTA_MAX_FLD_NUM				0xFF
	#define FSTA_MAX_FLD_LEN				0x3F
	#define FSTA_LEVEL(p)					((*p) & 0x40)
	#define FSTA_FLD_LEN(p)					((*p) & 0x3F)
	#define FSTA_FLD_NUM(p)					(*(p+1))
	#define FSTA_OVHD							2

	#define FOP_GET_FLD_FLAGS(p)			((*p) & 0x07)
	#define FOP_2BYTE_FLDNUM(bv)			((bv) & 0x02)
	#define FOP_2BYTE_FLDLEN(bv)			((bv) & 0x01)
	#define FOP_LOCAL_FLDNUM(bv)			((bv) & 0x04)


	#define FOP_TAGGED						0x80		// Use 4 left bits 1000 cxab
																// 0000 ffff tNum | LENGTH | VALUE

	#define FOP_IS_TAGGED(p)				((*(p) & 0xF0) == FOP_TAGGED)
	#define FTAG_LEVEL(p)					((*p) & 0x08)
	#define FTAG_GET_FLD_TYPE(c)			((c) & 0x0F)

	#define FOP_OPEN							0x90		// Use 4 left bits 1001 cxab
																// tNum | LENGTH | VALUE

	#define FOP_IS_OPEN(p)					((*(p) & 0xF0) == FOP_OPEN)
	#define FOPE_LEVEL(p)					((*p) & 0x08)


	#define FOP_SET_LEVEL					0xA0		// Use 5 left bits 1010 0vvv
	#define FOP_IS_SET_LEVEL(p)			((*(p) & 0xF8) == FOP_SET_LEVEL)
	#define FOP_LEVEL_MAX					0x07
	#define FSLEV_GET(p)						(*(p) & FOP_LEVEL_MAX)

	#define FOP_NO_VALUE						0xA8		// Use 5 left bits 1010 1ca0
																// a = 0		FLD_NUM=1 byte
																// a = 1		FLD_NUM=2 byte

	#define FOP_IS_NO_VALUE(p)				((*(p) & 0xF8) == FOP_NO_VALUE)
	#define FNOV_LEVEL(p)					((*p) & 0x04)
	#define FNOV_OVHD							2

	#define FOP_RECORD_INFO					0xB0		// Use 7 bits 1011 000b
																// LENGTH (1 or 2 bytes) | VALUE
	#define FOP_IS_RECORD_INFO(p) \
		((*(p) & 0xFE) == FOP_RECORD_INFO)

	#define DIN_KEY_SIZ						4
	#define ELM_DIN_OVHD						(BBE_KEY + DIN_KEY_SIZ)

	#define MAX_REC_ELM						250		// Max length of record portion
	#define MAX_FLD_OVHD						14			// Max field overhead

	// Extended field flags (for open and free fields)

	#define FOP_ENCRYPTED					0xE0		// Use 7 left bits 1110 000c ffff abab
																// tNum | LENGTH | eNum | eLENGTH | eVALUE

	#define FOP_IS_ENCRYPTED(p)			((*(p) & 0xFE) == FOP_ENCRYPTED)
	#define FENC_LEVEL(p)					((*p) & 0x01)
	#define FENC_FLD_TYPE(p)				(((*(p+1)) & 0xF0) >> 4)
	#define FENC_TAG_SZ(p)					(((*(p+1)) & 0x08) >> 3)
	#define FENC_LEN_SZ(p)					(((*(p+1)) & 0x04) >> 2)
	#define FENC_ETAG_SZ(p)					(((*(p+1)) & 0x02) >> 1)
	#define FENC_ELEN_SZ(p)					((*(p+1)) & 0x01)

	#define FOP_LARGE							0xD0

	/****************************************************************************
	Desc:
	****************************************************************************/
	inline FLMBOOL FOP_IS_LARGE(
		FLMBYTE *		pucFOP)
	{
		// 1101 xxec
		// fieldType (1 byte, 0000 ffff)
		// tagNum (2 bytes)
		// dataLen (4 bytes)
		//
		// If encrypted, the following are also present:
		//
		// encryptionId (2 bytes)
		// encryptionLength (4 bytes)

		if( (*pucFOP & 0xF0) == FOP_LARGE)
		{
			return( TRUE);
		}

		return( FALSE);
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	inline FLMBOOL FLARGE_LEVEL(
		FLMBYTE *		pucFOP)
	{
		flmAssert( FOP_IS_LARGE( pucFOP));

		if( (*pucFOP & 0x01))
		{
			return( TRUE);
		}

		return( FALSE);
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	inline FLMBOOL FLARGE_ENCRYPTED(
		FLMBYTE *		pucFOP)
	{
		flmAssert( FOP_IS_LARGE( pucFOP));

		if( (*pucFOP & 0x02))
		{
			return( TRUE);
		}

		return( FALSE);
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	inline FLMBYTE FLARGE_FLD_TYPE(
		FLMBYTE *		pucFOP)
	{
		flmAssert( FOP_IS_LARGE( pucFOP));
		return( *(pucFOP + 1) & 0x0F);
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	inline FLMUINT FLARGE_TAG_NUM(
		FLMBYTE *		pucFOP)
	{
		flmAssert( FOP_IS_LARGE( pucFOP));
		return( FB2UW( &pucFOP[ 2]));
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	inline FLMUINT FLARGE_DATA_LEN(
		FLMBYTE *		pucFOP)
	{
		flmAssert( FOP_IS_LARGE( pucFOP));
		return( FB2UD( &pucFOP[ 4]));
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	inline FLMUINT FLARGE_ETAG_NUM(
		FLMBYTE *		pucFOP)
	{
		flmAssert( FLARGE_ENCRYPTED( pucFOP));
		return( FB2UW( &pucFOP[ 8]));
	}

	/****************************************************************************
	Desc:
	****************************************************************************/
	inline FLMUINT FLARGE_EDATA_LEN(
		FLMBYTE *		pucFOP)
	{
		flmAssert( FLARGE_ENCRYPTED( pucFOP));
		return( FB2UD( &pucFOP[ 10]));
	}

	// SEN - Simple Encoded Number
	//
	// This is the variable length numbering system that can store
	// up to a 36 bit number in 1 to 5 bytes.
	// The SEN is the backbone to the index reference list compression
	// and the standard format for other functions that need to represent
	// a number in a variable number of bytes.

	#define SEN_1B_CODE	  0x00				// SEN 1 byte code
	#define SEN_1B_VAL	  127					// Max 1 byte value - 7 bits

	#define SEN_2B_CODE	  0x80				// SEN 2 byte code
	#define SEN_2B_CMSK	  0xC0				// Mask to check for code
	#define SEN_2B_VAL	  16383				// Max 2 byte value - 14 bits
	#define SEN_2B_MASK	  0x3F

	#define SEN_3B_CODE	  0xC0				// SEN 3 byte code
	#define SEN_3B_CMSK	  0xF0				// Mask to check for code
	#define SEN_3B_VAL	  1048575			// Max 3 byte value - 20 bits
	#define SEN_3B_MASK	  0x0F

	#define SEN_4B_CODE	  0xD0				// SEN 4 byte code
	#define SEN_4B_CMSK	  0xF0				// Mask to check for code
	#define SEN_4B_VAL	  268435455			// Max 4 byte value - 28 bits
	#define SEN_4B_MASK	  0x0F

	#define SEN_5B_CODE	  0xE0				// SEN 5 byte code
	#define SEN_5B_CMSK	  0xF0				// Mask to check for code
	#define SEN_5B_MASK	  0x0F

	#define SEN_FLAG		  0xF0				// Flag that contains some meaning

	#define SEN_MAX_SIZ	  7					// -2,000,000,000 is biggest SEN

	#define SEN_DOMAIN	  0xFC				// A domain in SEN format follows
	#define SEN_UPDATE_VER 0xFD				// Future


	// DIN - Dual Integer Numbers

	#define	DIN_ONE_RUN_LV 		0xF0		// Lowest value for one runs
	#define	DIN_MAX_1B_ONE_RUN	9			// Maximum one byte one run value
	#define	DIN_ONE_RUN_HV 		0xF8		// High value for one runs

	#define	DIN_IS_ONE_RUN(b) (((b)==1) || \
									(((b) >= DIN_ONE_RUN_LV) && ((b) <= DIN_ONE_RUN_HV)))

	#define	DIN_IS_REAL_ONE_RUN(b) \
									(((b) >= DIN_ONE_RUN_LV) && ((b) <= DIN_ONE_RUN_HV))


	// The reference set maximums are computed from the most bytes that
	// contain the maximum number of items that can exist within a
	// single domain.
	// The REF_SET_MAX_SIZ must contain more than one domain or the
	// set compression is not working as designed.	
	// The minimum value of REF_SET_MAX_SIZ should >= 180.  The worst pattern
	// is 02 01 02 01 02 01 = total must be > 256.	170 references + overhead
	//
	// REF_SET_FIRST_MAX is not used at this point.

	#define REF_SET_MAX_SIZ		180			// 170 references + extra stuff
	#define REF_SPLIT_50_50		50				// Be really conservative
	#define REF_SPLIT_90_10		0				// Split at first break

	#define SPLIT_90_10			0
	#define SPLIT_50_50			1

	// B-tree chain end indicator

	#define BT_END					((FLMUINT32)0xFFFFFFFF)

	// At the end of an element list

	#define ELEMENT_END			((FLMUINT16) 0xFFFF)

	// Domains are used for direct access to the index reference sets

	#define DIN_DOMAIN(din)		((din) >> 8)
	#define DRN_DOMAIN(drn)		((drn) >> 8)
	#define ZERO_DOMAIN			((FLMUINT) 0)
	#define MAX_DOMAIN			((FLMUINT) 0x1000000)

	// B-tree Block Scan Return Codes

	#define BT_EQ_KEY					0
	#define BT_GT_KEY					1
	#define BT_LT_KEY					2
	#define BT_END_OF_DATA			0xFFFF

	#define DRN_LAST_MARKER			((FLMUINT) 0xFFFFFFFF)
	#define DRN_LAST_MARKER_LEN	11

	// Block Header Layout

	#define BH_CHECKSUM_LOW 		0		// Low order bits of checksum.
													// Ver 3.0 low byte of blk address used
													// in the checksum value.
	#define BH_ADDR					0		// Block address
	#define BH_PREV_BLK				4		// Previous block in the chain
	#define BH_NEXT_BLK				8		// Next block in the chain
	#define BACKCHAIN_CNT			36		// Number of chains in a back chain

	#define BH_TYPE					12		// Block type - defined below
		#define BHT_FREE						0		// Free block - avail list
		#define BHT_LEAF						1		// Leaf block
		#define BHT_LFH_BLK					4		// LFH Header block
		#define BHT_PCODE_BLK				5		// PCODE block
		#define BHT_NON_LEAF					6		// Non-leaf block - variable key size
		#define BHT_NON_LEAF_DATA			7		// Non-leaf block data block - fixed key size
		#define BHT_NON_LEAF_COUNTS		8		// Non-leaf index with counts

	#define BHT_BI_BLK				0x30

	#define BH_GET_TYPE(blk) \
		(((blk)[BH_TYPE]) & 0x0F )
					
	#define BH_SET_BI(blk) \
		(((blk)[BH_TYPE]) |= BHT_BI_BLK)

	#define BH_UNSET_BI(blk) \
		(((blk)[BH_TYPE]) &= (~(BHT_BI_BLK)))

	#define BH_IS_BI(blk) \
		((((blk)[BH_TYPE]) & BHT_BI_BLK) == BHT_BI_BLK)

	#define BHT_ROOT_BLK				0x80

	#define BH_IS_ROOT_BLK(blk) \
		(((blk)[BH_TYPE]) & BHT_ROOT_BLK)

	#define BH_SET_ROOT_BLK(blk) \
		((blk)[BH_TYPE] |= BHT_ROOT_BLK)

	// The maximum levels in any b-tree be 8 levels with version 3.x. 
	// We will not worry about 2x compatibility problems until we understand
	// what needs to be done on the 2x to 30 conversion.
	// Very long keys in index records (500-1000 bytes) could easily run
	// out of 8 levels, but this is very unlikely at this time. 

	#define BH_LEVEL					13				// Block level (B-tree only)

	#define BH_MAX_LEVELS			8				// Max allowable b-tree levels
	#define MAX_LEVELS				BH_MAX_LEVELS

	#define BH_ELM_END				14				// End of the elements in a block
	#define BH_BLK_END				14				// End of the elements in a block

	#define BH_TRANS_ID				16				// Last transaction to update this block
	#define BH_PREV_TRANS_ID		20				// Previous transaction to update block
	#define BH_PREV_BLK_ADDR		24				// Pointer to previous image of blk
	#define BH_LOG_FILE_NUM			28				// Logical file number of block
	#define BH_ENCRYPTED				30				// Flag indicating if block is encrypted
	#define BH_CHECKSUM_HIGH		31				// High order bits of checksum

	#define BH_OVHD					32				// Overhead in the block header

	#define	BH_NEXT_BACKCHAIN	 	4				// Backchains of avail list - 4 bytes
	#define	BH_PREV_BACKCHAIN0	30				// Avail blocks as of version 3 not encrypted
	#define	BH_PREV_BACKCHAIN1 	13				// Prev backchain contains 3 bytes
	#define	BH_PREV_BACKCHAIN2 	28				// at different locations in the header
	#define	BH_PREV_BACKCHAIN3 	29				// Level and logical file num are used
	
	#define	BBE_LEM_LEN				3				// Length of leaf last element marker

	#define GET_BH_ADDR( pBlk) \
		(FB2UD(&(pBlk)[BH_ADDR]))

	#define SET_BH_ADDR( pBlk, dwAddr) \
		UD2FBA( dwAddr, &(pBlk)[BH_ADDR] )

	// Block access from the cache pointer

	#define GET_CABLKPTR(stack) \
		((stack)->pSCache->pucBlk)

	#define CABLK_ELM(stack,elm) \
		((stack)->pSCache->pucBlk[ (elm) ])

	// Block access from the pBlk

	#define SET_BLKPTR(stack) \
		((stack)->pBlk = stack->pSCache->pucBlk)

	#define BLK_PTR(stack) \
		((stack)->pBlk)

	#define BLK_ELM(stack,elm) \
		((stack)->pBlk[ (elm) ])

	#define BLK_ELM_ADDR(stack,elm) \
		(&((stack)->pBlk[ (elm) ]))

	#define CURRENT_ELM(stack) \
		(&((stack)->pBlk[ stack->uiCurElm ]))

	inline void flmCopyDrnKey(
		FLMBYTE *		pucDest,
		FLMBYTE *		pucSrc)
	{
#ifdef FLM_UNIX
		f_memcpy( pucDest, pucSrc, sizeof( FLMUINT32));
#else
		*((FLMUINT32 *)pucDest) = *((FLMUINT32 *)pucSrc);
#endif
	}

	// Resolving the block address into components.

	#define MAX_DATA_FILE_NUM_VER40	0x1FF
	#define MAX_LOG_FILE_NUM_VER40	0x3FF
	#define MAX_DATA_FILE_NUM_VER43	0x7FF
	#define MAX_LOG_FILE_NUM_VER43	0xFFF

	#define MAX_DATA_BLOCK_FILE_NUMBER(uiDbVersion) \
					(FLMUINT)(((uiDbVersion) >= FLM_FILE_FORMAT_VER_4_3) \
								 ? MAX_DATA_FILE_NUM_VER43 \
								 : MAX_DATA_FILE_NUM_VER40)

	#define FIRST_LOG_BLOCK_FILE_NUMBER(uiDbVersion) \
					(FLMUINT)(MAX_DATA_BLOCK_FILE_NUMBER(uiDbVersion) + 1)

	#define MAX_LOG_BLOCK_FILE_NUMBER(uiDbVersion) \
					(FLMUINT)(((uiDbVersion) >= FLM_FILE_FORMAT_VER_4_3) \
								 ? MAX_LOG_FILE_NUM_VER43 \
								 : MAX_LOG_FILE_NUM_VER40)

	#define FSGetFileNumber( uiBlkAddr) \
		((uiBlkAddr) & MAX_LOG_FILE_NUM_VER43)

	#define FSGetFileOffset( udBlkAddr) \
		((udBlkAddr) & 0xFFFFF000)

	#define FSBlkAddress( iFileNum, udFileOfs) \
		((udFileOfs) + (iFileNum))

	// Max file size and log threshold.

	#define MAX_FILE_SIZE_VER40			((FLMUINT)0x7FF00000)

	FINLINE FLMUINT flmGetMaxFileSize(
		FLMUINT		uiDbVersion,
		FLMBYTE *	pucLogHdr)
	{
		FLMUINT		uiMaxSize = MAX_FILE_SIZE_VER40;
		
		if( uiDbVersion >= FLM_FILE_FORMAT_VER_4_3)
		{
			uiMaxSize = ((FLMUINT)FB2UW( &pucLogHdr[ LOG_MAX_FILE_SIZE])) << 16;
			if( !uiMaxSize)
			{
				uiMaxSize = MAX_FILE_SIZE_VER40;
			}
		}
		
		return( uiMaxSize);
	}

	// Very large threshhold is the size we will allow the physical
	// log to grow to before we force a truncation.	 At the low end,
	// it is about 32 megabytes.  At the high end it is about
	// 1 gigabyte.

	#define LOW_VERY_LARGE_LOG_THRESHOLD_SIZE		((FLMUINT)0x2000000)
	#define HIGH_VERY_LARGE_LOG_THRESHOLD_SIZE	((FLMUINT)0x40000000)

	void ScaCleanupCache(
		FLMUINT				uiMaxLockTime);
	
	void ScaFreeModifiedBlocks(
		FDB *					pDb);
	
	FLMBOOL flmNeededByReadTrans(
		FFILE *				pFile,
		FLMUINT				uiLowTransId,
		FLMUINT				uiHighTransId);
	
	void ScaReleaseLogBlocks(
		FFILE *				pFile);
	
	RCODE ScaGetBlock(
		FDB *					pDb,
		LFILE *				pLFile,
		FLMUINT				uiBlkType,
		FLMUINT				uiBlkAddress,
		FLMUINT *			puiNumLooksRV,
		SCACHE **			ppSCacheRV);
	
	RCODE ScaCreateBlock(
		FDB *					pDb,
		LFILE *				pLFile,
		SCACHE **			ppSCacheRV);
	
	void ScaHoldCache(
		SCACHE *				pSCache);
	
	void ScaReleaseCache(
		SCACHE *				pSCache,
		FLMBOOL				bMutexAlreadyLocked);
	
	RCODE ScaLogPhysBlk(
		FDB *					pDb,
		SCACHE **			ppSCacheRV);
	
	RCODE ScaInit(
		FLMUINT				uiMaxSharedCache);
	
	RCODE ScaConfig(
		FLMUINT				uiType,
		void *				pvValue1,
		void *				pvValue2);
	
	void ScaExit( void);
	
	void ScaFreeFileCache(
		FFILE *				pFile);
	
	RCODE ScaDoCheckpoint(
		DB_STATS *			pDbStats,
		F_SuperFileHdl *	pSFileHdl,
		FFILE *				pFile,
		FLMBOOL				bTruncateRollBackLog,
		FLMBOOL				bForceCheckpoint,
		FLMINT				iForceReason,
		FLMUINT				uiCPFileNum,
		FLMUINT				uiCPOffset);
	
	RCODE ScaEncryptBlock(
		FFILE *				pFile,
		FLMBYTE *			pucBuffer,
		FLMUINT				uiBufLen,
		FLMUINT				uiBlockSize);
	
	RCODE ScaDecryptBlock(
		FFILE *				pFile,
		FLMBYTE *			pucBuffer);

	FLMUINT64 FSGetSizeInBytes(
		FLMUINT				uiMaxFileSize,
		FLMUINT				uiBlkAddress);
	
	RCODE FSGetBlock(
		FDB *					pDb,
		LFILE *				pLFile,
		FLMUINT				uiBlkAddress,
		BTSK *				pStack);
	
	void FSReleaseStackCache(
		BTSK *				pStack,
		FLMUINT				uiNumLevels,
		FLMBOOL				bMutexAlreadyLocked);
	
	RCODE FSBlockFree(
		FDB *					pDb,
		SCACHE *				pSCache);
	
	RCODE FSBlockFixLinks(
		FDB *					pDb,
		LFILE *				pLFile,
		SCACHE *				pSCache);
	
	RCODE FSBlockUseNextAvail(
		FDB *					pDb,
		LFILE *				pLFile,
		SCACHE **			ppSCacheRV);
	
	FLMUINT ALGetNBC(
		FLMBYTE *			pBlkBuf);
	
	#define ALGetNBC( pBlkBuf) \
		(FB2UD( &pBlkBuf [BH_NEXT_BACKCHAIN]))
	
	void ALPutNBC(
		FLMBYTE *			pBlkBuf,
		FLMUINT				uiAddr);
	
	#define ALPutNBC(pBlkBuf,uiAddr) \
		(UD2FBA( uiAddr, &(pBlkBuf)[ BH_NEXT_BACKCHAIN]))
	
	RCODE FSCombineBlks(
		FDB *					pDb,
		LFILE *				lfd,
		BTSK **				stackRV);
	
	FLMUINT FSBlkBuildPKC(
		BTSK *				stack,
		FLMBYTE *			pkcBuf,
		FLMUINT				uiFlags);
	
	#define FSBBPKC_BEFORE_CURELM			0
	#define FSBBPKC_AT_CURELM				1
	
	RCODE FSBlkMoveElms(
		BTSK *				newBlkStk,
		FLMBYTE *			inElm,
		FLMUINT				uiInsElmLen,
		FLMBYTE *			elmPKCBuf);
	
	inline FLMUINT FSRefFirst(
		BTSK *				stack,
		DIN_STATE * 		state,
		FLMUINT *			puiDomainRV);
	
	RCODE FSNextRecord(
		FDB *					pDb,
		LFILE *				pLFile,
		BTSK *				pStack);
	
	RCODE FSRefNext(
		FDB *					pDb,
		LFILE *				lfd,
		BTSK *				stk,
		DIN_STATE * 		state,
		FLMUINT *			puiDrnRV);
	
	RCODE FSRefSearch(
		BTSK *				stack,
		DIN_STATE * 		state,
		FLMUINT *			dinRV);
	
	FLMUINT DINNextVal(
		FLMBYTE *			dinPtr,
		DIN_STATE * 		state);
	
	FLMUINT SENNextVal(
		FLMBYTE ** 			senPtrRV);
	
	FLMUINT DINOneRunVal(
		FLMBYTE *			dinPtr,
		DIN_STATE * 		state);
	
	FLMUINT FSGetDomain(
		FLMBYTE **			curElmRV,
		FLMUINT				uiElmOvhd);
	
	RCODE FSBtPrevElm(
		FDB *					pDb,
		LFILE *				lfd,
		BTSK *				stack);
	
	FLMUINT FSRefLast(
		BTSK *				stack,
		DIN_STATE * 		state,
		FLMUINT *			domainRV);
	
	FLMUINT FSGetPrevRef(
		FLMBYTE *			pCurRef,
		DIN_STATE * 		pState,
		FLMUINT				uiTarget);
	
	RCODE FSRefPrev(
		FDB *					pDb,
		LFILE *				lfd,
		BTSK *				stk,
		DIN_STATE * 		state,
		FLMUINT *			drnRV);
	
	RCODE FSBtDelete(
		FDB *					pDb,
		LFILE *				lfd,
		BTSK **				stack);
	
	RCODE FSDelParentElm(
		FDB *					pDb,
		LFILE *				lfd,
		BTSK **				stackRV);
	
	RCODE FSNewLastBlkElm(
		FDB *					pDb,
		LFILE *				logDef,
		BTSK **				stackRV,
		FLMUINT				uiFlags);
	
	#define FSNLBE_GREATER			0x01
	#define FSNLBE_LESS				0x02
	#define FSNLBE_POSITION			0x04
	
	RCODE FSBlkDelElm(
		BTSK *				stack);
	
	void FSSetChildBlkAddr(
		FLMBYTE *			childElmPtr,
		FLMUINT				uiBlkAddr,
		FLMUINT				uiElmOvhd);
	
	RCODE FSBtReplace(
		FDB *					pDb,
		LFILE *				lfd,
		BTSK **				stackRV,
		FLMBYTE *			elm,
		FLMUINT				uiElmLen);
	
	RCODE FSBtInsert(
		FDB *					pDb,
		LFILE *				lfd,
		BTSK **				stackRV,
		FLMBYTE *			elm,
		FLMUINT				uiElmLen);
	
	FLMUINT FSSetElmOvhd(
		FLMBYTE *			elm,
		FLMUINT				uiElmOvhd,
		FLMUINT				pkc,
		FLMUINT				uiKeyLen,
		FLMBYTE *			byteOneAddr);
	
	RCODE FSReadRecord(
		FDB *					pDb,
		LFILE *				pLFile,
		FLMUINT				drn,
		FlmRecord **		ppRecord,
		FLMUINT *			puiRecTransId,
		FLMBOOL *			pbMostCurrent);
	
	RCODE FSReadElement(
		FDB *					pDb,
		F_Pool *				pPool,
		LFILE *				pLFile,
		FLMUINT				drn,
		BTSK *				pStack,
		FLMBOOL				bOkToPreallocSpace,
		FlmRecord **		ppRecord,
		FLMUINT *			puiRecTransId,
		FLMBOOL *			pbMostCurrent);
	
	RCODE FSRecUpdate(
		FDB *					pDb,
		LFILE *				lfd,
		FlmRecord * 		pRecord,
		FLMUINT				drn,
		FLMUINT				uiAddAppendFlags);
	
		#define REC_UPD_NEW_RECORD			2
		#define REC_UPD_ADD					1
		#define REC_UPD_MODIFY				0
		#define REC_UPD_DELETE				0
	
	RCODE FSGetNextDrn(
		FDB *					pDb,
		LFILE *				lfd,
		FLMBOOL				bUpdateNextDrn,
		FLMUINT *			drnRV);
	
	RCODE FSSetNextDrn(
		FDB *					pDb,
		BTSK *				stack,
		FLMUINT				drn,
		FLMBOOL				bManditory);
	
	RCODE FSRefUpdate(
		FDB *					pDb,
		LFILE *				pLFile,
		KREF_ENTRY * 		pKref);
	
	void FSFreeIxCounts(
		FDB * 				pDb);
	
	RCODE FSCommitIxCounts(
		FDB * 				pDb);
	
	RCODE FSUpdateBlkCounts(
		FDB *					pDb,
		BTSK *				pStack,
		FLMUINT				uiNewCount);
	
	RCODE FSUpdateAdjacentBlkCounts(
		FDB *					pDb,
		LFILE *				pLFile,
		BTSK *				pStack,
		BTSK *				pNextBlkStk);
	
	RCODE FSChangeCount(
		FDB *					pDb,
		BTSK *				pStack,
		FLMBOOL				bAddReference);
	
	RCODE FSChangeBlkCounts(
		FDB *					pDb,
		BTSK *				pStack,
		FLMINT				iDelta);
	
	RCODE FSGetBtreeRefPosition(
		FDB *					pDb,
		BTSK *				pStack,
		DIN_STATE *			pDinState,
		FLMUINT *			puiRefPosition);
	
	RCODE FSPositionSearch(
		FDB *					pDb,
		LFILE *				pLFile,
		FLMUINT				uiRefPosition,
		BTSK * *				ppStack,
		FLMUINT *			puiRecordId,
		FLMUINT *			puiDomain,
		DIN_STATE *			pDinState);
	
	RCODE FSPositionScan(
		BTSK *				pStack,
		FLMUINT				uiRelativePosition,
		FLMUINT *			puiRelativePosInElement,
		FLMUINT *			puiRecordId,
		FLMUINT *			puiDomain,
		DIN_STATE *			pDinState);
	
	RCODE FSPositionToRef(
		BTSK *				pStack,
		FLMUINT				uiRelativePosition,
		FLMUINT *			puiRecordId,
		FLMUINT *			puiDomain,
		DIN_STATE *			pDinState);
		
	RCODE FSSetInsertRef(
		FLMBYTE *			dest,
		FLMBYTE *			src,
		FLMUINT				drn,
		FLMUINT *			puiSetLenRV );
	
	RCODE FSSetDeleteRef(
		FLMBYTE *			dest,
		FLMBYTE *			src,
		FLMUINT				drn,
		FLMUINT *			puiSetLenRV);
	
	FLMUINT SENValLen(
		FLMBYTE *			senPtr);
	
	#define	SENValLen(ptr) \
		(SENLenArray[ *(ptr) >> 4 ])
	
	FLMUINT SENPutNextVal(
		FLMBYTE ** 			senPtrRV,
		FLMUINT				senValue );
	
	FLMUINT DINPutOneRunVal(
		FLMBYTE *			dinPtr,
		DIN_STATE * 		state,
		FLMUINT				value);
	
	RCODE FSRefSplit(
		FDB *					pDb,
		LFILE *				lfd,
		BTSK **				stkRV,
		FLMBYTE *			elmBuf,
		FLMUINT				drn,
		FLMUINT				uiDeleteFlag,
		FLMUINT				uiSplitFactor);
	
	RCODE FSBtSearch(
		FDB *					pDb,
		LFILE *				pLFile,
		BTSK **				ppStackRV,
		FLMBYTE *			pKey,
		FLMUINT				uiKeyLen,
		FLMUINT				uiDrnDomain);
	
	RCODE FSBtSearchEnd(
		FDB *					pDb,
		LFILE *				pLFile,
		BTSK **				pStackRV,
		FLMUINT				uiDrn);
	
	RCODE FSGetRootBlock(
		FDB *					pDb,
		LFILE **				ppLFile,
		LFILE *				pTmpLFile,
		BTSK *				pStack);
	
	RCODE FSBtScan(
		BTSK *				stk,
		FLMBYTE *			key,
		FLMUINT				uiKeyLen,
		FLMUINT				uiDrnDomain);
	
	RCODE FSBtScanNonLeafData(
		BTSK *				pStack,
		FLMUINT				uiDrn);
	
	void FSBlkToStack(
		BTSK *				stack);
	
	RCODE FSBtScanTo(
		BTSK *				stk,
		FLMBYTE *			key,
		FLMUINT				uiKeyLen,
		FLMUINT				drnDomain);
	
	RCODE FSBlkNextElm(
		BTSK *				stack);
	
	RCODE FSBtNextElm(
		FDB *					pDb,
		LFILE *				pLFile,
		BTSK *				pStack);
	
	RCODE FSAdjustStack(
		FDB *					pDb,
		LFILE *				pLFile,
		BTSK *				stack,
		FLMBOOL				bMovedNext);
	
	RCODE FSBlkSplit(
		FDB *					pDb,
		LFILE *				pLFile,
		BTSK **				stkRV,
		FLMBYTE *			elm,
		FLMUINT				uiElmLen);
	
	RCODE dbLock(
		FDB *					pDb,
		FLMUINT				uiMaxLockWait);
	
	RCODE dbUnlock(
		FDB *					pDb);
	
	RCODE flmLFileInit(
		FDB *					pDb,
		LFILE *				pLFile);
	
	RCODE flmLFileRead(
		FDB *					pDb,
		LFILE *				pLFile);
	
	RCODE flmBufferToLFile(
		FLMBYTE *			pBuf,
		LFILE *				pLFile,
		FLMUINT				uiBlkAddress,
		FLMUINT				uiOffsetInBlk);
	
	RCODE flmLFileWrite(
		FDB *					pDb,
		LFILE *				pLFile);
	
	RCODE flmLFileCreate(
		FDB *					pDb,
		LFILE *				pLFile,
		FLMUINT				uiLfNum,
		FLMUINT				uiLfType);
	
	RCODE flmLFileDictUpdate(
		FDB *					pDb,
		LFILE **				ppDictLFile,
		FLMUINT *			puiDrnRV,
		FlmRecord * 		pNewDictRecord,
		FlmRecord * 		pOldDictRecord,
		FLMBOOL				bDoInBackground,
		FLMBOOL				bCreateSuspended,
		FLMBOOL *			pbLogCompleteIndexSet,
		FLMBOOL				bRebuildOp = FALSE);
	
	RCODE FSComputeRecordBlocks(
		BTSK *				pFromStack,
		BTSK *				pUntilStack,
		FLMUINT *			puiLeafBlocksBetween,
		FLMUINT *			puiTotalRecords,		
		FLMBOOL *			pbTotalsEstimated);
	
	RCODE FSComputeIndexCounts(
		BTSK *				pFromStack,
		BTSK *				pUntilStack,
		FLMUINT *			puiLeafBlocksBetween,
		FLMUINT *			puiTotalKeys,
		FLMUINT *			puiTotalRefs,
		FLMBOOL *			pbTotalsEstimated);
	
	FLMUINT FSElementRefCount(
		BTSK *				pStack);
	
	RCODE FSBlockCounts(
		BTSK *				pStack,
		FLMUINT				uiFirstElement,
		FLMUINT				uiLastElement,
		FLMUINT *			puiFirstKeyCount,
		FLMUINT *			puiElementCount,
		FLMUINT *			puiRefCount);
	
	RCODE flmWriteLogHdr(
		DB_STATS *			pDbStats,
		F_SuperFileHdl *	pSFileHdl,
		FFILE *				pFile,
		FLMBYTE *			pucLogHdr,
		FLMBYTE *			pucCPLogHdr,
		FLMBOOL				bIsCheckpoint);
	
	RCODE flmPhysRollback(
		FDB *					pDb,
		FLMUINT				uiLogEOF,
		FLMUINT				uiFirstLogBlkAddr,
		FLMBOOL				bDoingRecovery,
		FLMUINT				uiMaxTransID);
	
	RCODE lgFlushLogBuffer(
		DB_STATS *			pDbStats,
		F_SuperFileHdl *	pSFileHdl,
		FFILE *				pFile);
	
	RCODE lgOutputBlock(
		DB_STATS *			pDbStats,
		F_SuperFileHdl *	pSFileHdl,
		FFILE *				pFile,
		SCACHE *				pLogBlock,
		FLMBYTE *			pucBlk,
		FLMUINT *			puiLogEofRV);
	
	void lgSetSyncCheckpoint(
		FFILE *				pFile,
		FLMUINT				uiCheckpoint,
		FLMUINT				uiBlkAddress);
	
	FLMUINT lgHdrCheckSum(
		FLMBYTE *			pucLogHdr,
		FLMBOOL				bCompare);
	
	RCODE FSVersionConversion40(
		FDB *					pDb,
		FLMUINT				uiNewVersion,
		STATUS_HOOK 		fnStatusCallback,
		void *				pvUserData);

	RCODE FSFlushElement(
		FDB *					pDb,
		LFILE *				pLFile,
		UCUR *				updCur);
	
	/****************************************************************************
	Desc:	Get the previous backchain (PBC) address given an block
	****************************************************************************/
	inline FLMUINT ALGetPBC(
		FLMBYTE *	pucBlkBuf)
	{
		FLMUINT		uiPbcAddr;
	
		uiPbcAddr  = ((FLMUINT) pucBlkBuf [BH_PREV_BACKCHAIN1]) << 24;
		uiPbcAddr |= ((FLMUINT) pucBlkBuf [BH_PREV_BACKCHAIN2]) << 16;
		uiPbcAddr |= ((FLMUINT) pucBlkBuf [BH_PREV_BACKCHAIN3]) << 8;
		uiPbcAddr |= (FLMUINT) pucBlkBuf [BH_PREV_BACKCHAIN0];
	
		return( uiPbcAddr);
	}
	
	/****************************************************************************
	Desc:	Get the previous backchain (PBC) address given an block
	****************************************************************************/
	inline void ALPutPBC(
		FLMBYTE *	pucBlkBuf,
		FLMUINT		uiAddr)
	{
		pucBlkBuf [BH_PREV_BACKCHAIN1] = (FLMBYTE) (uiAddr >> 24);
		pucBlkBuf [BH_PREV_BACKCHAIN2] = (FLMBYTE) (uiAddr >> 16);
		pucBlkBuf [BH_PREV_BACKCHAIN3] = (FLMBYTE) (uiAddr >>	 8);
	
		// Code doesn't support old pre 3.0 format
		pucBlkBuf [BH_PREV_BACKCHAIN0] = (FLMBYTE) uiAddr;
	}
	
	/****************************************************************************
	Desc:	Free Chain - reset the avail block with zeros
	****************************************************************************/
	inline void ALResetAvailBlk(
		FLMBYTE *		pucBlkBuf)
	{
		UD2FBA( 0, &pucBlkBuf [BH_NEXT_BACKCHAIN]);
	
		// This is ok to set the [0] backchain - doubles as encryption value.
		pucBlkBuf [BH_PREV_BACKCHAIN0] =
		pucBlkBuf [BH_PREV_BACKCHAIN1] =
		pucBlkBuf [BH_PREV_BACKCHAIN2] =
		pucBlkBuf [BH_PREV_BACKCHAIN3] = 0;
	}
	
	/****************************************************************************
	Desc:		Compare 2 PKC buffers
	Return:	Number of bytes that were equal (from left to right)
	****************************************************************************/
	inline FLMUINT FSElmComparePKC(
		FLMBYTE *		pPkcBuf1,
		FLMUINT			uiPkcBufLen1,
		FLMBYTE *		pPkcBuf2,
		FLMUINT			uiPkcBufLen2)
	{
		FLMUINT			uiMinBytes = f_min( uiPkcBufLen1, uiPkcBufLen2);
		FLMUINT			uiEqualBytes = 0;
	
		while( uiMinBytes--)
		{
			if( *pPkcBuf1++ != *pPkcBuf2++ )
				break;
			uiEqualBytes++;
		}
		return( uiEqualBytes);
	}
	
	/****************************************************************************
	Desc:		Returns the parent element's child block value
	Return:	Address of child block
	****************************************************************************/
	inline FLMUINT FSChildBlkAddr(
		BTSK *			pStack)
	{
		FLMBYTE *		childBlkPtr;
		FLMUINT			uiElmOvhd = pStack->uiElmOvhd;
				
		if( uiElmOvhd == BNE_KEY_START || uiElmOvhd == BNE_KEY_COUNTS_START)
		{
			childBlkPtr = BLK_ELM_ADDR( pStack, pStack->uiCurElm + BNE_CHILD_BLOCK );
			return FB2UD( childBlkPtr);
		}
		else if( uiElmOvhd == BNE_DATA_OVHD)
		{
			childBlkPtr = BLK_ELM_ADDR( pStack, pStack->uiCurElm + BNE_DATA_CHILD_BLOCK );
			return FB2UD( childBlkPtr);
		}
		else
		{
			// Pre 3.x format is no longer supported.
			// Corruption
			flmAssert( 0);
			return BNE_KEY_START;
		}
	}
	
	/****************************************************************************
	Desc:		Release the current block in the 'stack' 
	Out:		pStack->pBlk, pSCache are set to NULL values.
	Notes:	Supports a NULL block (defined as pSCache == NULL)
	****************************************************************************/
	inline void FSReleaseBlock(
		BTSK *		pStack,				// Stack of variables for each level
		FLMBOOL		bMutexAlreadyLocked)
	{
		// Release the current block, if any
		
		if( pStack->pSCache)
		{
			ScaReleaseCache( pStack->pSCache, bMutexAlreadyLocked);
			pStack->pSCache = NULL;
			pStack->pBlk = NULL;

			// NOTE: Do NOT unset pStack->uiBlkAddr.	There are cases where we
			// will release the block, but come back later and re-get it using
			// the block address that is in the stack.
		}
	}
	
	/****************************************************************************
	Desc:		Log the current block.	The pointer to the block buffer may
				change on the log call.
	Out:		pStack->pBlk may be changed
	****************************************************************************/
	inline RCODE FSLogPhysBlk(
		FDB *			pDb,
		BTSK *		pStack)
	{
		RCODE			rc;
			
		if( RC_OK( rc = ScaLogPhysBlk( pDb, &pStack->pSCache)))
		{
			pStack->pBlk = pStack->pSCache->pucBlk;
		}
		else
		{
			ScaReleaseCache( pStack->pSCache, FALSE);
			pStack->pBlk = NULL;
			pStack->pSCache = NULL;
		}
		
		return( rc);
	}
	
	/****************************************************************************
	Desc: Initialize a stack array for cache access.  Set all of the pSCache
			pointers in the array to NULL.  This will prevent them from being
			released if they were never filled with anything.
	****************************************************************************/
	inline void FSInitStackCache(
		BTSK *		pStack,
		FLMUINT		uiNumLevels)
	{
		while( uiNumLevels--)
		{
			pStack->pSCache = NULL;
			pStack->pBlk = NULL;
			pStack->uiBlkAddr = BT_END;
			pStack++;
		}
	}
	
	/****************************************************************************
	Desc: Returns TRUE if a 3x address is less than another address.
			This will also work with 2x address.
	****************************************************************************/
	inline FLMBOOL FSAddrIsBelow(
		FLMUINT			uiAddress1,
		FLMUINT			uiAddress2)
	{
		if( FSGetFileNumber( uiAddress1) == FSGetFileNumber( uiAddress2))
		{
			if( FSGetFileOffset( uiAddress1) >= FSGetFileOffset( uiAddress2))
			{
				return FALSE;
			}
		}
		else if( FSGetFileNumber( uiAddress1) > FSGetFileNumber( uiAddress2))
		{
			return FALSE;
		}
		return TRUE;
	}
	
	/****************************************************************************
	Desc: Returns TRUE if a 3x address is less than or equal another address.
			This will also work with 2x address.
	****************************************************************************/
	inline FLMBOOL FSAddrIsAtOrBelow(
		FLMUINT			uiAddress1,
		FLMUINT			uiAddress2)
	{
		if( FSGetFileNumber( uiAddress1) == FSGetFileNumber( uiAddress2))
		{
			if( FSGetFileOffset( uiAddress1) > FSGetFileOffset( uiAddress2))
			{
				return FALSE;
			}
		}
		else if( FSGetFileNumber( uiAddress1) > FSGetFileNumber( uiAddress2))
		{
			return FALSE;
		}
		return TRUE;
	}
	
	/****************************************************************************
	Desc:		Put the next DIN value - high level without one run worries
	Out:		value put into dinPtr[state[0]]
	Return:	length of DIN
	****************************************************************************/
	inline FLMUINT DINPutNextVal(
		FLMBYTE *		dinPtr,
		DIN_STATE *		state,
		FLMUINT			value)
	{
		FLMUINT			uiLength;
		
		dinPtr += state->uiOffset;
		uiLength = SENPutNextVal( &dinPtr, value);
		state->uiOffset += uiLength;
		
		return( uiLength );
	}
	
	/****************************************************************************
	Desc: This routine gets the number of bytes to encrypt for a block.
			For encrypted blocks (new to FLM_FILE_FORMAT_VER_4_6), it must return
			the block end rounded up to the next 16 byte boundary.  For
			non-encrypted blocks, in order to maintain compatibility on
			a database that has been converted, we must return the block
			end rounded up to the next 4 byte boundary.	This is because
			when a database is converted to 4.60, we don't go through
			and recalculate the checksums on every block to go to 16 byte
			boundaries.	 Already-existing blocks will have been calcuated
			to a four byte boundary.  We can check BH_ENCRYPTED for new
			blocks that are on the 16 byte boundary, because that flag was
			never set prior to version 4.60, and in 4.60+ that is the only
			type of block where it will be set to a 16 byte boundary.
	Ret:	encryption size - FLMUINT
	****************************************************************************/
	inline FLMUINT getEncryptSize(
		FLMBYTE *	pBlk)
	{
		FLMUINT	uiLen = (FLMUINT)FB2UW( &pBlk [BH_ELM_END]);
		if (!pBlk [BH_ENCRYPTED])
		{
			if (uiLen % sizeof( FLMUINT32) != 0)
			{
				uiLen += (FLMUINT)(sizeof( FLMUINT32) - (uiLen % sizeof( FLMUINT32)));
			}
		}
		else if (uiLen < BH_OVHD)
		{
			uiLen = BH_OVHD;
		}
		else
		{
			if (uiLen % 16)
			{
				uiLen += (FLMUINT)(16 - (uiLen % 16));
			}
		}
		return( uiLen);
	}
	
	/****************************************************************************
	Desc:		return the first DRN in an elements reference list
	In:		BTSK * stack, state - should be DIN_STATE_SIZ
				* puiDomain - returns the elements domain
	Out:		state information updated to refer to the last reference & puiDomain
	Return:	DIN the din of the first reference
	****************************************************************************/
	inline FLMUINT FSRefFirst(
		BTSK *		pStack,			// Small stack to hold b-tree variables
		DIN_STATE * pState,			// Holds offset, one run number, etc.
		FLMUINT *	puiDomain)		// Returns the elements domain
	{
		FLMBYTE *	pCurElm = CURRENT_ELM( pStack);
		
		// Point past the domain, ignore return value
		
		*puiDomain = FSGetDomain( &pCurElm, pStack->uiElmOvhd);
	
		RESET_DINSTATE_p( pState);
	
		// Don't use DIN because state must be set to zero after getting value
		
		return( SENNextVal( &pCurElm));
	}
		
	/****************************************************************************
	Desc:
	****************************************************************************/
	typedef struct UCUR
	{
		BTSK *		pStack;									// Points to current stack level
		FLMUINT		uiDrn;									// Domain Record Number
		FLMUINT		uiBufLen;								// Length of the buffer
		FLMUINT		uiUsedLen;								// Used length in the buffer
		FLMUINT		uiFlags;									// Bit flags for values below
	#define UCUR_REPLACE			1							// Replace current element
	#define UCUR_INSERT			2							// Insert current element
	#define UCUR_LAST_TIME		4							// Set on last insert/replace
		FLMBYTE		pKeyBuf[ DIN_KEY_SIZ ];				// Holds the DIN key
		FLMBYTE		pElmBuf[ ELM_DIN_OVHD + 256];		// Holds each element
	} UCUR;
	
#include "fpackoff.h"

#endif
