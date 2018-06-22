//-------------------------------------------------------------------------
// Desc:	Typedefs for strucures needed to build pcode.
// Tabs:	3
//
// Copyright (c) 1991-1992, 1995-2007 Novell, Inc. All Rights Reserved.
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

#ifndef FDICT_H
#define FDICT_H

#include "fpackon.h"
// IMPORTANT NOTE: No other include files should follow this one except
// for fpackoff.h

// Logical File Save Area Layout for 4.x files.

#define LFH_LF_NUMBER_OFFSET	0		// Logical file number
#define LFH_TYPE_OFFSET			2	 	// Type of logical file
#define LFH_STATUS_OFFSET		3		// Contains status bits
#define LFH_ROOT_BLK_OFFSET	4		// B-TREE root block address
//#define LFH_FUTURE1			8		// Not necessarily zeroes - Code bases
												// 31 and 40 put stuff here.
#define LFH_NEXT_DRN_OFFSET	12		// Next DRN for containers
#define LFH_MAX_FILL_OFFSET	16		// Max fill % after rightmost split.
#define LFH_MIN_FILL_OFFSET	17		// Min fill % in blk after normal delete
//#define LFH_FUTURE2			18		// Filled with zeros
#define LFH_SIZE					32		// Maximum size of LFH.

#define FFILE_MIN_FILL			35
#define FFILE_MAX_FILL			91

struct TDICT;
struct LFILE;

RCODE fdictRebuild(
	FDB *					pDb);

RCODE	fdictBuildTables(
	TDICT *				pTDict,
	FLMBOOL				bRereadLFiles,
	FLMBOOL				bNewDict);

RCODE fdictInitTDict(
	FDB *					pDb,
	TDICT *				pTDict);

RCODE fdictCopySkeletonDict(
	FDB *					pDb);

RCODE fdictCloneDict(
	FDB *					pDb);

RCODE fdictFixupLFileTbl(
	FDICT *				pDict);

RCODE fdictProcessAllDictRecs( 
	FDB *					pDb,
	TDICT *				pTDict);

RCODE fdictProcessRec( 
	TDICT *				pTDict,
	FlmRecord *			pRecord,
	FLMUINT				uiDictRecNum);

RCODE DDGetFieldType(  
	FlmRecord *			pRecord,
	void *				pvField,
	FLMUINT *			puiFldInfo);

RCODE DDGetEncType(
	FlmRecord *			pRecord,
	void *				pvField,
	FLMUINT *			puiFldInfo);

RCODE fdictCreateNewDict(
	FDB *					pDb);

RCODE  fdictCreate( 
	FDB *					pDb,
	const char *		pszDictPath,
	const char *		pDictBuf);

RCODE flmAddRecordToDict( 
	FDB *					pDb,
	FlmRecord *			pRecord,
	FLMUINT				uiDictId,
	FLMBOOL				bRereadLFiles);

/****************************************************************************
Desc:	Structure for type, DRN and name for data dictionary entries
****************************************************************************/
typedef struct DDENTRY
{
	DDENTRY * 	pNextEntry;						
	void *		vpDef;	
	FLMUINT  	uiEntryNum;			
	FLMUINT   	uiType;				
} DDENTRY;

/****************************************************************************
Desc:	Temporary field info used during a database create or dictionary
		modification.  This field is pointed to by the DDEntry structure.
****************************************************************************/
typedef struct TFIELD
{
	FLMUINT		uiFldNum;
	FLMUINT		uiFldInfo;
} TFIELD;

/****************************************************************************
Desc:	Temporary encryption definition info used during a database create or
		dictionary modification.  This field is pointed to by the
		DDEntry structure.
****************************************************************************/
typedef struct TENCDEF
{
	FLMUINT		uiRecNum;
	FLMUINT		uiState;
	FLMUINT		uiAlgType;
	FLMBYTE *	pucKeyInfo;
	FLMUINT		uiLength;
} TENCDEF;

/****************************************************************************
Desc:		Used as temporary storage for index definitions during a
			database create or dictionary modification.  This field is
			pointed to by the DDEntry structure.
****************************************************************************/
typedef struct TIFP
{
	TIFP *		pNextTIfp;			// Linked list of IFPs							
	FLMBOOL		bFieldInThisDict;	// Was field reference found in the
											//	dictionary we are updating? 
	FLMUINT		uiFldNum;			// Fixedup field ID value						
} TIFP;

/****************************************************************************
Desc:		Used as temporary storage for index definitions during a
			database create or dictionary modification.  This field is
			pointed to by the DDEntry structure.
****************************************************************************/
typedef struct TIFD
{
	TIFP * 		pTIfp;		  		// Linked list of temporary field paths	
	TIFD *		pNextTIfd;			// Linked List										
	FLMUINT		uiFlags;				// Field type & processing flags				
	FLMUINT		uiNextFixupPos;	// Next fixup position  
	FLMUINT		uiLimit;				// Zero or limit of characters/bytes		
	FLMUINT		uiCompoundPos;		// Position of this field is in 
											//	the compound key.  Zero based number.	
} TIFD;

/****************************************************************************
Desc:		Used as temporary storage for index definitions during a
			database create or dictionary modification.  This field is
			pointed to by the DDEntry structure.
****************************************************************************/
typedef struct TIXD
{
	TIFD *   	pNextTIfd;  		// Linked list of TIFDs							
	FLMUINT		uiFlags;				// Index attributes								
	FLMUINT		uiContainerNum;	// Container number of data records			
	FLMUINT		uiNumFlds;			// Number of field definitions				
	FLMUINT		uiLanguage;			// Index language
	FLMUINT		uiEncId;				// Encryption Definition
} TIXD;

/****************************************************************************
Desc:		Contains the dictionary entries through parsing all of the dictionary 
			records. Used for expanding record definitions, checking index 
			definitions, building fixup position values and last of all 
			BUILDING THE PCODE.
****************************************************************************/
typedef struct TDICT
{
	FDB *			pDb;
	F_Pool     	pool;					// Pool for the DDENTRY allocations.
	LFILE *		pLFile;				// Dictionary container LFile
	FDICT *		pDict;				// Pointer to new dictionary.
	FLMBOOL		bWriteToDisk;		// Flag indicating if PCODE should be
											//	written to disk after being generated.

	// Variables for building dictionaries 

	FLMUINT		uiCurPcodeAddr;	// Current pcode block we are adding to
	FLMUINT		uiBlockSize;		// PCODE Block size

	// Used in building the temporary structures 
	
	FLMUINT		uiVersionNum;		// Version number of database.
	DDENTRY *	pFirstEntry;
	DDENTRY *	pLastEntry;

	FLMUINT		uiNewIxds;
	FLMUINT		uiNewIfds;
	FLMUINT		uiNewFldPaths;
	FLMUINT		uiNewLFiles;

	FLMUINT		uiTotalItts;
	FLMUINT		uiTotalIxds;
	FLMUINT		uiTotalIfds;
	FLMUINT		uiTotalFldPaths;
	FLMUINT		uiTotalLFiles;

	FLMUINT		uiBadField;			// Set to field number on most errors.
	FLMUINT		uiBadReference;	// Same

	FLMUINT		uiDefaultLanguage;// Default language to set in each index.
} TDICT;

/****************************************************************************
Desc:		A Item Type consists of a byte that describes the type of item
			like a field, index or container.
			For fields a ITT will also indicate the fields delete status.
****************************************************************************/
typedef struct ITT
{
	FLMUINT		uiType;
	void *		pvItem;		// Points to LFILE if index or container
									// If field, is NULL or points to first IFD.
} ITT;

// Bit values for uiType.  The 4 low bits contain the field type.
// See FLM_XXXX_TYPE in FLAIM.H for lower four bits.

#define ITT_FLD_GET_TYPE( pItt)		(((pItt)->uiType) & 0x0F)
#define ITT_FLD_IS_INDEXED( pItt)	(((pItt)->pvItem) ? TRUE : FALSE)
#define ITT_FLD_GET_STATE( pItt)		(((pItt)->uiType) & 0x30)

#define ITT_FLD_STATE_MASK			0x30
#define ITT_FLD_STATE_ACTIVE		0x00 	// Normal active field
#define ITT_FLD_STATE_CHECKING	0x10	// Field has been marked to be checked
#define ITT_FLD_STATE_UNUSED		0x30	// Field is not used.
#define ITT_FLD_STATE_PURGE		0x20	// Purge this field from the database.
											// And delete the dictionary definition

#define ITT_ENC_STATE_MASK			0x30
#define ITT_ENC_STATE_ACTIVE		0x00 	// Normal active field
#define ITT_ENC_STATE_CHECKING	0x10	// EncDef has been marked to be checked
#define ITT_ENC_STATE_UNUSED		0x30	// EncDef is not used.
#define ITT_ENC_STATE_PURGE		0x20	// EncDef record is being deleted.  Decrypt the
													// encrypted field as it can no longer be
													// encrypted.

#define ITT_ENCDEF_TYPE			0xAF	// Encrypted Definition Record
#define ITT_INDEX_TYPE			0xBF
#define ITT_CONTAINER_TYPE		0xCF
#define ITT_EMPTY_SLOT 			0xEF
#define ITT_INFO_MASK			0x0F

#define ITT_IS_FIELD(pItt)	(((pItt)->uiType & ITT_INFO_MASK) != ITT_INFO_MASK)
#define ITT_IS_CONTAINER(pItt)	((pItt)->uiType == ITT_CONTAINER_TYPE)
#define ITT_IS_INDEX(pItt)			((pItt)->uiType == ITT_INDEX_TYPE)
#define ITT_IS_ENCDEF(pItt)		((pItt)->uiType == ITT_ENCDEF_TYPE)

/****************************************************************************
Desc:		This structure holds the information for an index definition.
			There may be multiple IXDs for the same index number.
****************************************************************************/
typedef struct IXD
{
	FLMUINT		uiIndexNum;				// Index number.
	FLMUINT		uiContainerNum;		// Container number being indexed.
	IFD *			pFirstIfd;				// Points to first IFD
	FLMUINT		uiNumFlds;		  		// Number of index fields in the IFD.
	FLMUINT		uiFlags;
		#define IXD_UNIQUE				0x00001	// Unique index
		#define IXD_COUNT					0x00002	// Count keys and references
		#define IXD_EACHWORD				0x00100	// FUTURE: FLAIMs fulltext indexing.
		#define IXD_HAS_POST				0x01000	// Has post keys parts.
		#define IXD_HAS_SUBSTRING		0x02000
		#define IXD_POSITIONING			0x04000	// The index has positioning counts.
		#define IXD_OFFLINE				0x08000
		#define IXD_SUSPENDED			0x10000

	FLMUINT		uiLanguage;				// WP.LRS language number (not code!)
		#define US_LANG			0
		#define DEFAULT_LANG		US_LANG

#define	TRANS_ID_OFFLINE			TRANS_ID_HIGH_VALUE
#define	TRANS_ID_ALWAYS_ONLINE	TRANS_ID_LOW_VALUE

	FLMUINT		uiLastContainerIndexed;	// Last container indexed if index
													// covers multiple containers.
	FLMUINT		uiLastDrnIndexed;		// If value is not DRN_LAST_MARKER then 
												// update index with keys from a record 
												// update if drn of record is <= of 
												// this value.
	FLMUINT		uiEncId;					// The ID / Drn of the Encryption record (if used)
} IXD;

/****************************************************************************
Desc:		This structure contains an index field definition.
****************************************************************************/
typedef struct IFD
{
	FLMUINT		uiFldNum;				// Field being indexed.
	FLMUINT		uiIndexNum;				// Index number.
	IXD *			pIxd;						// IXD corresponding to wIndexNum
	FLMUINT		uiFlags;					// The first 4 bits contain field type
												// Use FLM_XXXXX_TYPE definitions.

	IFD *			pNextInChain;			// Next IFD in the chain that has this
												// field number and is used in another index.
	FLMUINT *	pFieldPathCToP;		// Child to parent field path (zero term)
	FLMUINT *	pFieldPathPToC;		// Parent to child field path (zero term)

	FLMUINT		uiLimit;					// Zero or # of characters/bytes to limit.
#define IFD_DEFAULT_LIMIT					256
#define IFD_DEFAULT_SUBSTRING_LIMIT		48

	FLMUINT		uiCompoundPos;			// Position of this field is in
												// the compound key.  Zero based number.
} IFD;

#define IFD_GET_FIELD_TYPE(pIfd)	((pIfd)->uiFlags & 0x0F)
#define IFD_SET_FIELD_TYPE(pIfd,type)	((pIfd)->uiFlags = ((pIfd)->uiFlags & 0xFFFFFFF0) | (type))
#define IFD_FIELD	 			0x00000010	// There must always be some value
#define IFD_VALUE				0x00000010	// Value agrees with parsing syntax

#define IFD_EACHWORD			0x00000020	// Index each and every word in the field
#define IFD_CONTEXT			0x00000040	// Index the tag and NOT the value
#define IFD_COMPOUND			0x00000080	// Index multiple fields

#define IFD_POST				0x00000100	// Place case info at end of compound key
#define IFD_UPPER	 			0x00000200	// Uppercase keys only
#define IFD_OPTIONAL			0x00000400	// This field is optional (compound)
													// Phasing this value out.

// Note: the unique flag is for future compatiblity.

#define IFD_UNIQUE_PIECE	0x00000800	// Better name

#define IFD_REQUIRED_PIECE	0x00001000	// Required piece (not optional)
#define IFD_REQUIRED_IN_SET 0x0002000	// Required within a set of fields.

#define IFD_LAST				0x00008000	// Last IFD for this index definition

#define IFD_SUBSTRING		0x00040000	// Index all substrings pieces
#define IFD_DRN				0x00080000	// index DRN value
#define IFD_FIELDID_PAIR	0x00200000	// Data | fieldID pair.
#define IFD_MIN_SPACES		0x00400000	// Removed leading/trailing spaces.
													// Combine multiple spaces into 1 space.
													// Minimize spaces
#define IFD_NO_SPACE			0x00800000	// Remove all spaces
#define IFD_NO_DASH			0x01000000	// Remove all dashes
#define IFD_NO_UNDERSCORE	0x02000000	// Change underscores to spaces,
													// Must be applied before nospace/minspace
#define IFD_ESC_CHAR			0x04000000	// Placehold so that a query can parse the input
													// string and find a literal '*' or '\\'.

#define IFD_IS_POST_TEXT(pIfd)		(((pIfd)->uiFlags & IFD_POST) && \
												(IFD_GET_FIELD_TYPE(pIfd) == FLM_TEXT_TYPE))
#define IFD_DEFAULT_LIMIT					256
#define IFD_DEFAULT_SUBSTRING_LIMIT		48

/**************************************************************************
Desc: 	This structure is a header for a FLAIM dictionary.  All of
			the information in this structure is static.
**************************************************************************/
typedef struct FDICT
{
	FDICT *		pNext;			// Pointer to next FDICT structure in the list,
										// if any.  All versions of a dictionary that
										// are currently in use are linked together.
										// Usually, there will be only one local
										// dictionary in the list.
	FDICT *		pPrev;			// Previous FDICT structure in the list.
	FFILE *		pFile;			// File this dictionary is associated with.
										// A null value means it is not yet linked
										// to a file.
	FLMUINT		uiDictSeq;		// This is the sequence number of the dictionary

	// Local Dictionary Tables.

	LFILE *		pLFileTbl;		// Logical file (index or container)
	FLMUINT		uiLFileCnt;
#define LFILE_DATA_CONTAINER_OFFSET			0
#define LFILE_DICT_CONTAINER_OFFSET			1
#define LFILE_DICT_INDEX_OFFSET				2
#define LFILE_TRACKER_CONTAINER_OFFSET		3

	ITT *			pIttTbl;
	FLMUINT		uiIttCnt;

	IXD *			pIxdTbl;
	FLMUINT		uiIxdCnt;

	IFD *			pIfdTbl;
	FLMUINT		uiIfdCnt;

	FLMUINT *	pFldPathsTbl;
	FLMUINT		uiFldPathsCnt;

	FLMUINT		uiUseCount;		// Number of FDB structures currently
										// pointing to this dictionary.
} FDICT;

/****************************************************************************
Desc:		This is a temporary structure that is used when building compound
			keys.
****************************************************************************/
typedef struct CDL
{
	void *		pField;			// Field to be included in a compound key
	void *		pRootContext;	// Points to root context of field path
	CDL *			pNext;			// Pointer to the next CDL entry.
} CDL;

/****************************************************************************
Desc:		This keeps track of the logical file information for an index or
			a container.
****************************************************************************/
typedef struct LFILE
{
	FLMUINT	   uiRootBlk;				// Address of root block.
	FLMUINT		uiNextDrn;				// Next DRN - only use when root is null
	FLMUINT		uiBlkAddress;			// Block address of LFile entry.
	FLMUINT		uiOffsetInBlk;			// Offset within block of entry.
	FLMUINT		uiLfNum;					// Index number or container number.
	FLMUINT		uiLfType; 				// Type of logical file. */
	FLMBOOL		bMakeFieldIdTable;	// Boolean that indicates whether or not
												// for this container when we create
												// records in cache we should create a
												// field id table for the level-1 fields.
	IXD *			pIxd;						// If an index, points to the IXD.

} LFILE;

#include "fpackoff.h"

#endif
