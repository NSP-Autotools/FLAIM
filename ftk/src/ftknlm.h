//-------------------------------------------------------------------------
// Desc:	Toolkit - cross platform APIs for system functionality.
// Tabs:	3
//
// Copyright (c) 1991-2007 Novell, Inc. All Rights Reserved.
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
//-------------------------------------------------------------------------

#ifndef FTKNLM_H
#define FTKNLM_H

#ifdef FLM_NLM
#pragma pack(push,1) 
	
	#if defined( FLM_WATCOM_NLM)
		#pragma warning 007 9

		// Disable "Warning! W549: col(XX) 'sizeof' operand contains
		// compiler generated information"
		
		#pragma warning 549 9
		
		// Disable "Warning! W656: col(XX) define this function inside its class
		// definition (may improve code quality)"
		
		#pragma warning 656 9
		
		// Disable Warning! W555: col(XX) expression for 'while' is always
		// "false"
		
		#pragma warning 555 9
	#endif
	
	#ifdef FLM_LIBC_NLM
	
		#define _POSIX_SOURCE

		#include <stdio.h>
		#include <string.h>
		#include <pthread.h>
		#include <unistd.h>
		#include <errno.h>
		#include <library.h>
		#include <fcntl.h>
		#include <sys/stat.h>
		#include <sys/unistd.h>
		#include <glob.h>
		#include <netware.h>
		#include <semaphore.h>
		#include <malloc.h>
		#include <novsock2.h>
	
		#ifndef LONG
			#define LONG	unsigned long
		#endif

	#endif

	void * f_getNLMHandle( void);
	
	#ifdef FLM_RING_ZERO_NLM
		typedef unsigned long 								LONG;
		typedef unsigned short 								WORD;
		typedef unsigned char 								BYTE;
		typedef unsigned short   							wchar_t;
		typedef unsigned char 								wsnchar;
		typedef unsigned int 								BOOL;
		typedef unsigned int 								DWORD;
		typedef unsigned int *								LPDWORD;
		typedef unsigned long 								ULONG;
		typedef unsigned char								UCHAR;
		typedef DWORD 											WPARAM;
		typedef DWORD 											LPARAM;
		typedef unsigned int									UINT;
		typedef unsigned int 								size_t;
		typedef void * 										MUTEX;
		typedef void * 										SEMAPHORE;
		typedef unsigned long 								ERROR;
		typedef void * 										SPINLOCK;
		typedef LONG 											clockAndStatus[3];
		typedef unsigned int									SOCKET;
		typedef unsigned long 								f_va_list;
		typedef SEMAPHORE										F_SEM;
		typedef unsigned long  								mode_t;
		typedef void *											scr_t;
		typedef void *											rtag_t;
		typedef FLMUINT32										uint32_t;
		typedef FLMUINT16										uint16_t;
		typedef FLMBYTE										uint8_t;
	
		typedef void (* F_EXIT_FUNC)( void);
		
		#define TimerSignature								0x524D4954
		#define SemaphoreSignature							0x504D4553
		#define AllocSignature								0x54524C41
		#define ScreenSignature								0x4E524353
		
		#define DOSNameSpace									0
		#define MACNameSpace									1
		#define MacNameSpace									MACNameSpace
		#define NFSNameSpace									2
		#define FTAMNameSpace								3
		#define OS2NameSpace									4
		#define LONGNameSpace								4
		#define NTNameSpace									5
		#define MAX_NAMESPACES								6
		
		#define NO_RIGHTS_CHECK_ON_OPEN_BIT				0x00010000
		#define ALLOW_SECURE_DIRECTORY_ACCESS_BIT		0x00020000
		#define READ_ACCESS_BIT								0x00000001
		#define WRITE_ACCESS_BIT							0x00000002
		#define DENY_READ_BIT								0x00000004
		#define DENY_WRITE_BIT								0x00000008
		#define COMPATABILITY_MODE_BIT					0x00000010
		#define FILE_WRITE_THROUGH_BIT					0x00000040
		#define FILE_READ_THROUGH_BIT						0x00000080
		#define ALWAYS_READ_AHEAD_BIT						0x00001000
		#define NEVER_READ_AHEAD_BIT						0x00002000
		
		#define READ_ONLY_BIT								0x00000001
		#define HIDDEN_BIT									0x00000002
		#define SYSTEM_BIT									0x00000004
		#define EXECUTE_BIT									0x00000008
		#define SUBDIRECTORY_BIT							0x00000010
		#define ARCHIVE_BIT									0x00000020
		#define SHAREABLE_BIT								0x00000080
		#define OLD_PRIVATE_BIT								0x00000080
		#define NO_SUBALLOC_BIT								0x00000800
		#define SMODE_BITS									0x00000700
		#define TRANSACTION_BIT		  						0x00001000
		#define READ_AUDIT_BIT								0x00004000
		#define WRITE_AUDIT_BIT		  						0x00008000
		#define IMMEDIATE_PURGE_BIT	  					0x00010000
		#define RENAME_INHIBIT_BIT							0x00020000
		#define DELETE_INHIBIT_BIT							0x00040000
		#define COPY_INHIBIT_BIT							0x00080000
		#define FILE_AUDITING_BIT							0x00100000
		#define REMOTE_DATA_ACCESS_BIT					0x00400000 
		#define REMOTE_DATA_INHIBIT_BIT					0x00800000
		#define REMOTE_DATA_SAVE_KEY_BIT					0x01000000
		#define COMPRESS_FILE_IMMEDIATELY_BIT			0x02000000
		#define DATA_STREAM_IS_COMPRESSED_BIT			0x04000000
		#define DO_NOT_COMPRESS_FILE_BIT					0x08000000
		#define CANT_COMPRESS_DATA_STREAM_BIT			0x20000000
		#define ATTR_ARCHIVE_BIT							0x40000000
		#define ZFS_VOLATILE_BIT							0x80000000
		
		#define VOLUME_AUDITING_BIT	  					0x01
		#define SUB_ALLOCATION_ENABLED_BIT				0x02
		#define FILE_COMPRESSION_ENABLED_BIT			0x04
		#define DATA_MIGRATION_ENABLED_BIT				0x08
		#define NEW_TRUSTEE_COUNT_BIT						0x10
		#define DIR_SVCS_OBJ_UPGRADED_BIT				0x20
		#define VOLUME_IMMEDIATE_PURGE_ENABLED_BIT	0x40
		
		#define PrimaryDataStream							0
		#define MACResourceForkDataStream				1
		#define FTAMDataStream								2
		
		#define DefinedAccessRightsBits					0x01FB
		#define MaximumDirectoryAccessBits				0x01FF
		#define AllValidAccessBits							0x100001FF
		
		#define SYNCCLOCK_CLOCK_BIT						0x00000001L
		#define SYNCCLOCK_TICK_INCREMENT_BIT			0x00000002L
		#define SYNCCLOCK_ADJUSTMENT_BIT					0x00000004L
		#define SYNCCLOCK_GROSS_CORRECTION_BIT			0x00000008L
		#define SYNCCLOCK_ADJUSTMENT_COUNT_BIT			0x00000010L
		#define SYNCCLOCK_STATUS_BIT						0x00000020L
		#define SYNCCLOCK_STD_TICK_BIT					0x00000040L
		#define SYNCCLOCK_EVENT_TIME_BIT					0x00000080L
		#define SYNCCLOCK_EVENT_OFFSET_BIT				0x00000100L
		#define SYNCCLOCK_HARDWARE_CLOCK_BIT			0x00000200L
		#define SYNCCLOCK_RESERVED1_BIT					0x00000400L
		#define SYNCCLOCK_DAYLIGHT_BIT					0x00000800L
		#define SYNCCLOCK_TIMEZONE_OFFSET_BIT			0x00001000L
		#define SYNCCLOCK_TZNAME_BIT						0x00002000L
		#define SYNCCLOCK_TIMEZONE_STR_BIT				0x00004000L
		#define SYNCCLOCK_DAYLIGHT_OFFSET_BIT			0x00008000L
		#define SYNCCLOCK_DAYLIGHT_ON_OFF_BIT			0x00010000L
		#define SYNCCLOCK_START_DST_BIT					0x00020000L
		#define SYNCCLOCK_STOP_DST_BIT					0x00040000L
		#define SYNCCLOCK_ALL_DEFINED_BITS		   	(0x0007FFFFL & ~SYNCCLOCK_HARDWARE_CLOCK_BIT)
		
		#define LDModuleIsReEntrantBit					0x00000001
		#define LDModuleCanBeMultiplyLoadedBit			0x00000002
		#define LDSynchronizeStart							0x00000004
		#define LDPseudoPreemptionBit						0x00000008
		#define LDLoadInKernel								0x00000010
		#define Available_0									0x00000020
		#define LDAutoUnload  								0x00000040
		#define LDHiddenModule								0x00000080
		#define LDDigitallySignedFile						0x00000100
		#define LDLoadProtected								0x00000200
		#define LDSharedLibraryModule						0x00000400
		#define LDRestartable								0x00000800
		#define LDUnsafeToUnloadNow						0x00001000
		#define LDModuleIsUniprocessor					0x00002000
		#define LDPreemptable								0x00004000
		#define LDHasSystemCalls							0x00008000
		#define LDVirtualMemory								0x00010000
		#define LDAllExportsMTSafe							0x00020000
		
		#define DFSFailedCompletion           			-1
		#define DFSNormalCompletion           			0
		#define DFSInsufficientSpace          			1
		#define DFSVolumeSegmentDeactivated   			4
		#define DFSTruncationFailure          			16
		#define DFSHoleInFileError            			17
		#define DFSParameterError             			18
		#define DFSOverlapError               			19
		#define DFSSegmentError               			20
		#define DFSBoundryError               			21
		#define DFSInsufficientLimboFileSpace 			22
		#define DFSNotInDirectFileMode        			23
		#define DFSOperationBeyondEndOfFile   			24
		#define DFSOutOfHandles               			129
		#define DFSHardIOError                			131
		#define DFSInvalidFileHandle          			136
		#define DFSNoReadPrivilege            			147
		#define DFSNoWritePrivilege           			148
		#define DFSFileDetached               			149
		#define DFSInsufficientMemory         			150
		#define DFSInvalidVolume              			152
		#define DFSIOLockError                			162
	
		#define MModifyNameBit                 		0x0001
		#define MFileAttributesBit             		0x0002
		#define MCreateDateBit                 		0x0004
		#define MCreateTimeBit                 		0x0008
		#define MOwnerIDBit                    		0x0010
		#define MLastArchivedDateBit           		0x0020
		#define MLastArchivedTimeBit           		0x0040
		#define MLastArchivedIDBit             		0x0080
		#define MLastUpdatedDateBit            		0x0100
		#define MLastUpdatedTimeBit            		0x0200
		#define MLastUpdatedIDBit              		0x0400
		#define MLastAccessedDateBit           		0x0800
		#define MInheritanceRestrictionMaskBit 		0x1000
		#define MMaximumSpaceBit               		0x2000
		#define MLastUpdatedInSecondsBit       		0x4000
	
		#define MAX_NETWARE_VOLUME_NAME					16
		#define F_NW_DEFAULT_VOLUME_NUMBER				0
		
		#define LO_RETURN_HANDLE        					0x00000040
		
		#define CURSOR_NORMAL								0x0C0B
		#define CURSOR_THICK									0x0C09
		#define CURSOR_BLOCK									0x0C00
		#define CURSOR_TOP									0x0400
	
		#define htonl											WS2_32_htonl 
		#define ntohl											WS2_32_ntohl 
		#define htons											WS2_32_htons 
		#define ioctlsocket									WS2_32_ioctlsocket
		#define ntohs											WS2_32_ntohs 
		#define send											WS2_32_send
		#define recv											WS2_32_recv
		#define bind											WS2_32_bind
		#define listen											WS2_32_listen
		#define closesocket									WS2_32_closesocket
		#define getpeername									WS2_32_getpeername
		#define getsockname									WS2_32_getsockname
		#define getsockopt									WS2_32_getsockopt
		#define select											WS2_32_select
		#define setsockopt									WS2_32_setsockopt
		#define socket											WS2_32_socket
		#define inet_addr										WS2_32_inet_addr
		#define inet_ntoa										WS2_32_inet_ntoa
		#define gethostbyaddr								WS2_32_gethostbyaddr
		#define gethostbyname								WS2_32_gethostbyname
		#define gethostname									WS2_32_gethostname
		
		#define connect(s,name,namelen) \
			WSAConnect(s,name,namelen, 0,0,0,0)
			
		#define accept(s,addr,addrlen) \
			WSAAccept(s,addr,addrlen,0,0)
		
		#define IPPROTO_IP									0						// dummy for IP
		#define IPPROTO_ICMP									1						// control message protocol
		#define IPPROTO_IGMP									2						// internet group management protocol
		#define IPPROTO_GGP									3						// gateway (deprecated)
		#define IPPROTO_TCP									6						// tcp
		#define IPPROTO_PUP									12						// pup
		#define IPPROTO_UDP									17						// user datagram protocol
		#define IPPROTO_IDP									22						// xns idp
		#define IPPROTO_ND									77						// UNOFFICIAL net disk proto
	
		#define TCP_NODELAY     							0x0001
	
		#define SOCK_STREAM     							1						// stream socket
		#define SOCK_DGRAM      							2						// datagram socket
		#define SOCK_RAW        							3						// raw-protocol interface
		#define SOCK_RDM        							4						// reliably-delivered message
		#define SOCK_SEQPACKET  							5						// sequenced packet stream
	
		#define WSADESCRIPTION_LEN							256
		#define WSASYS_STATUS_LEN							128
		
		#define s_addr  										S_un.S_addr
		#define s_host  										S_un.S_un_b.s_b2
		#define s_net   										S_un.S_un_b.s_b1
		#define s_imp   										S_un.S_un_w.s_w2
		#define s_impno 										S_un.S_un_b.s_b4
		#define s_lh    										S_un.S_un_b.s_b3
	
		#define h_addr  										h_addr_list[0]
		
		#define FD_SETSIZE      							64
		
		#define WINSOCK_VERSION 							MAKEWORD(2,2)
	
		#define INVALID_SOCKET  							(SOCKET)(~0)
		#define SOCKET_ERROR            					(-1)
		
		#define AF_INET         							2
	
		#define WSABASEERR              					10000
		#define WSAEINTR                					(WSABASEERR+4)
		#define WSAEBADF                					(WSABASEERR+9)
		#define WSAEACCES               					(WSABASEERR+13)
		#define WSAEFAULT               					(WSABASEERR+14)
		#define WSAEINVAL               					(WSABASEERR+22)
		#define WSAEMFILE               					(WSABASEERR+24)
		#define WSAEWOULDBLOCK          					(WSABASEERR+35)
		#define WSAEINPROGRESS          					(WSABASEERR+36)
		#define WSAEALREADY             					(WSABASEERR+37)
		#define WSAENOTSOCK             					(WSABASEERR+38)
		#define WSAEDESTADDRREQ         					(WSABASEERR+39)
		#define WSAEMSGSIZE             					(WSABASEERR+40)
		#define WSAEPROTOTYPE           					(WSABASEERR+41)
		#define WSAENOPROTOOPT          					(WSABASEERR+42)
		#define WSAEPROTONOSUPPORT      					(WSABASEERR+43)
		#define WSAESOCKTNOSUPPORT      					(WSABASEERR+44)
		#define WSAEOPNOTSUPP           					(WSABASEERR+45)
		#define WSAEPFNOSUPPORT         					(WSABASEERR+46)
		#define WSAEAFNOSUPPORT         					(WSABASEERR+47)
		#define WSAEADDRINUSE           					(WSABASEERR+48)
		#define WSAEADDRNOTAVAIL        					(WSABASEERR+49)
		#define WSAENETDOWN             					(WSABASEERR+50)
		#define WSAENETUNREACH          					(WSABASEERR+51)
		#define WSAENETRESET            					(WSABASEERR+52)
		#define WSAECONNABORTED         					(WSABASEERR+53)
		#define WSAECONNRESET           					(WSABASEERR+54)
		#define WSAENOBUFS              					(WSABASEERR+55)
		#define WSAEISCONN              					(WSABASEERR+56)
		#define WSAENOTCONN             					(WSABASEERR+57)
		#define WSAESHUTDOWN            					(WSABASEERR+58)
		#define WSAETOOMANYREFS         					(WSABASEERR+59)
		#define WSAETIMEDOUT            					(WSABASEERR+60)
		#define WSAECONNREFUSED         					(WSABASEERR+61)
		#define WSAELOOP                					(WSABASEERR+62)
		#define WSAENAMETOOLONG         					(WSABASEERR+63)
		#define WSAEHOSTDOWN            					(WSABASEERR+64)
		#define WSAEHOSTUNREACH         					(WSABASEERR+65)
		#define WSAENOTEMPTY            					(WSABASEERR+66)
		#define WSAEPROCLIM             					(WSABASEERR+67)
		#define WSAEUSERS               					(WSABASEERR+68)
		#define WSAEDQUOT               					(WSABASEERR+69)
		#define WSAESTALE               					(WSABASEERR+70)
		#define WSAEREMOTE              					(WSABASEERR+71)
		#define WSASYSNOTREADY          					(WSABASEERR+91)
		#define WSAVERNOTSUPPORTED      					(WSABASEERR+92)
		#define WSANOTINITIALISED       					(WSABASEERR+93)
		#define WSAEDISCON              					(WSABASEERR+101)
		#define WSAENOMORE              					(WSABASEERR+102)
		#define WSAECANCELLED           					(WSABASEERR+103)
		#define WSAEINVALIDPROCTABLE    					(WSABASEERR+104)
		#define WSAEINVALIDPROVIDER     					(WSABASEERR+105)
		#define WSAEPROVIDERFAILEDINIT  					(WSABASEERR+106)
		#define WSASYSCALLFAILURE       					(WSABASEERR+107)
		#define WSASERVICE_NOT_FOUND    					(WSABASEERR+108)
		#define WSATYPE_NOT_FOUND       					(WSABASEERR+109)
		#define WSA_E_NO_MORE           					(WSABASEERR+110)
		#define WSA_E_CANCELLED					         (WSABASEERR+111)
		#define WSAEREFUSED             					(WSABASEERR+112)
		#define WSAHOST_NOT_FOUND       					(WSABASEERR+1001)
		#define WSATRY_AGAIN            					(WSABASEERR+1002)
		#define WSANO_RECOVERY          					(WSABASEERR+1003)
		#define WSANO_DATA              					(WSABASEERR+1004)
	
		#define INADDR_ANY									0x00000000
		#define INADDR_LOOPBACK								0x7f000001
		#define INADDR_BROADCAST							0xffffffff
		#define INADDR_NONE									0xffffffff
		#define ADDR_ANY										INADDR_ANY
	
		/*************************************************************************
		Desc:
		*************************************************************************/
		typedef struct LoadDefinitionStructure
		{
			struct LoadDefinitionStructure *		LDLink;
			struct LoadDefinitionStructure *		LDKillLink;
			struct LoadDefinitionStructure *		LDScanLink;
			struct ResourceTagStructure *			LDResourceList;
			LONG 											LDIdentificationNumber;
			LONG 											LDCodeImageOffset;
			LONG 											LDCodeImageLength;
			LONG 											LDDataImageOffset;
			LONG 											LDDataImageLength;
			LONG 											LDUninitializedDataLength;
			LONG 											LDCustomDataOffset;
			LONG 											LDCustomDataSize;
			LONG 											LDFlags;
			LONG 											LDType;
			LONG (*LDInitializationProcedure)(
					struct LoadDefinitionStructure *	LoadRecord,
					struct ScreenStruct *				screenID,
					BYTE *									CommandLine,
					BYTE *									loadDirectoryPath,
					LONG 										uninitializedDataLength,
					LONG 										fileHandle,
					LONG (*ReadRoutine)(
							LONG 					fileHandle,
							LONG 					offset,
							void *				buffer,
							LONG 					numberOfBytes),
					LONG 										customDataOffset,
					LONG 										customDataSize);
					
			void (*LDExitProcedure)(void);
			
			LONG (*LDCheckUnloadProcedure)(
					struct ScreenStruct *screenID);
					
			void *									LDPublics;
			BYTE 										LDFileName[36];
			BYTE 										LDName[128];
			LONG *									LDCLIBLoadStructure;
			LONG *									LDNLMDebugger;
			LONG 										LDParentID;
			LONG 										LDReservedForCLIB;
			void *									AllocMemory;
			LONG 										LDTimeStamp;
			void *									LDModuleObjectHandle;
			LONG 										LDMajorVersion;
			LONG										LDMinorVersion;
			LONG 										LDRevision;
			LONG 										LDYear;
			LONG 										LDMonth;
			LONG 										LDDay;
			BYTE *									LDCopyright;
			LONG 										LDSuppressUnloadAllocMsg;
			LONG 										Reserved2;
			LONG 										Reserved3;
			LONG 										Reserved4[64];
			LONG 										Reserved5[12];
			LONG 										Reserved6;
			void *									LDDomainID;
			struct LoadDefinitionStructure *	LDEnvLink;
			void *									LDAllocPagesListHead;
			void *									LDTempPublicList;
			LONG 										LDMessageLanguage;
			BYTE **									LDMessages;
			LONG 										LDMessageCount;
			BYTE *									LDHelpFile;
			LONG 										LDMessageBufferSize;
			LONG 										LDHelpBufferSize;
			LONG 										LDSharedCodeOffset;
			LONG 										LDSharedCodeLength;
			LONG										LDSharedDataOffset;
			LONG 										LDSharedDataLength;
			LONG (*LDSharedInitProcedure)(
					struct LoadDefinitionStructure *	LoadRecord,
					struct ScreenStruct *				screenID,
					BYTE *									CommandLine);
					
			void (*LDSharedExitProcedure)(void);
			
			LONG 										LDRPCDataTable;
			LONG 										LDRealRPCDataTable;
			LONG 										LDRPCDataTableSize;
			LONG 										LDNumberOfReferencedPublics;
			void **									LDReferencedPublics;
			LONG 										LDNumberOfReferencedExports;
			LONG 										LDNICIObject;
			LONG 										LDAllocPagesListLocked;
			void *									LDAddressSpace;
			LONG 										Reserved7;
			void *									MPKStubAddress;
			LONG	 									MPKStubSize;
			LONG 										LDBuildNumber;
			void *									LDExtensionData;
		} LoadDefinitionStructure;
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		typedef struct FCBType
		{
			LONG												OpenFileLink;
			LONG												StationLink;
			struct FCBType	*								ShareLinkNext;
			struct FCBType *								ShareLinkLast;
			struct OwnerRestrictionNodeStructure *	OwnerRestrictionNode;
			struct SpaceRestrictionNodeStructure *	SubdirectoryRestrictionNode;
			LONG												OpenCount;
			LONG												DirectoryEntry;
			LONG												DirectoryNumber;
			LONG												ActualDirectoryEntry;
			LONG												FileSize;
			LONG												FirstCluster;
			LONG												CurrentBlock;
			LONG												CurrentCluster;
			struct FATStruct *							FATTable;
			LONG												TNodePointer;
			LONG												DateValueStamp;
			void *											TurboFAT;
			LONG												OldFileSize;
			LONG												TransactionPointer;
			LONG												Station;
			LONG												Task;
			BYTE												HandleCount[4];
			BYTE												Flags;
			BYTE												TTSFlags;
			BYTE												ByteToBlockShiftFactor;
			BYTE												BlockToSectorShiftFactor;
			BYTE												SectorToBlockMask;
			BYTE												VolumeNumber;
			BYTE												ExtraFlags;
			BYTE												DataStream;
			LONG												CommitSemaphore;
			LONG												ActualOldLastCluster;
			LONG												FCBInUseCount;
			BYTE												ExtraExtraFlags;
			BYTE												SubAllocFlags;
			BYTE												RemoveSkipUpdateFlags;
			BYTE												DeCompressFlags;
			LONG												VolumeManagerID;
			LONG												SubAllocSemaphore;
			LONG												SAStartingSector[2];
			LONG												SANumberOfSectors[2];
			LONG												SAFATCount;
			LONG												TempCacheListHead;
			LONG												TempCacheListTail;
			struct CompressControlNodeStructure	*	CompressControlNode;
			struct FCBType *								DeCompressFCB;
			LONG												DeCompressPosition;
			LONG												DeCompressHandle;
			LONG												RALastReadStartOffset;
			LONG												RALastReadEndOffset;
			LONG												RANextReadAheadOffset;
			LONG												RAHalfSize;
			LONG												MoreFlags;
			LONG												StationBackLink;
			SPINLOCK											DFSSpinLock;
			LONG												DFSUseCount;
			LONG												DFSCurrentCluster;
			LONG												DFSCurrentBlock;
			LONG												unused;
			LONG												unused1;
			LONG												unused2;
			LONG												unused3;
		} FCBType;
	
		/*************************************************************************
		Desc:
		*************************************************************************/
		struct DirectoryStructure
		{
		#define NumberOfDirectoryTrustees	4
			int 	DSubdirectory;
			LONG	DFileAttributes;
			BYTE	DUniqueID;
			BYTE	DFlags;
			BYTE	DNameSpace;
			BYTE	DFileNameLength;
			BYTE	DFileName[12];
			LONG	DCreateDateAndTime;
			LONG	DOwnerID;
			LONG	DLastArchivedDateAndTime;
			LONG	DLastArchivedID;
			LONG	DLastUpdatedDateAndTime;
			LONG	DLastUpdatedID;
			LONG	DFileSize;
			LONG	DFirstBlock;
			LONG	DNextTrusteeEntry;
			LONG	DTrustees[NumberOfDirectoryTrustees];
			LONG	DLookUpEntryNumber;
			int 	DLastUpdatedInSeconds;
			WORD	DTrusteeMask[NumberOfDirectoryTrustees];
			WORD	DChangeReferenceID;
			WORD	DLastAccessedTime;
			WORD	DMaximumAccessMask;
			WORD	DLastAccessedDate;
			LONG	DDeletedFileTime;
			LONG	DDeletedDateAndTime;
			LONG	DDeletedID;
			LONG	DExtendedAttributes;
			LONG	DDeletedBlockSequenceNumber;
			LONG	DPrimaryEntry;
			LONG	DNameList;
		};
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		typedef struct Synchronized_Clock_T
		{
		#define MAX_TIME_ZONE_STRING_LENGTH		80
			LONG	clock[2];
			LONG	statusFlags;
			LONG	adjustmentCount;
			LONG	adjustment[2];
			LONG	grossCorrection[2];
			LONG	tickIncrement[2];
			LONG	stdTickIncrement[2];
			LONG	eventOffset[2];
			LONG	eventTime;
			LONG	daylight;
			long	timezoneOffset;
			long	tzname[2];
			char	timeZoneString[MAX_TIME_ZONE_STRING_LENGTH];
			long	daylightOffset;
			long	daylightOnOff;
			LONG	startDSTime;
			LONG	stopDSTime;
		} Synchronized_Clock_T;
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		typedef struct
		{
			char **							ppszArgV;
			char *							pszArgs;
			char *							pszThreadName;
			int								iArgC;
			LoadDefinitionStructure *	moduleHandle;
		} ARG_DATA;
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		struct VolumeInformationStructure
		{
			LONG VolumeAllocationUnitSizeInBytes;
			LONG VolumeSizeInAllocationUnits;
			LONG VolumeSectorSize;
			LONG AllocationUnitsUsed;
			LONG AllocationUnitsFreelyAvailable;
			LONG AllocationUnitsInDeletedFilesNotAvailable;
			LONG AllocationUnitsInAvailableDeletedFiles;
			LONG NumberOfPhysicalSegmentsInVolume;
			LONG PhysicalSegmentSizeInAllocationUnits[64];
		};
	
		/*************************************************************************
		Desc:
		*************************************************************************/
		struct ModifyStructure
		{
			BYTE *MModifyName;
			LONG  MFileAttributes;
			LONG  MFileAttributesMask;
			WORD  MCreateDate;
			WORD  MCreateTime;
			LONG  MOwnerID;
			WORD  MLastArchivedDate;
			WORD  MLastArchivedTime;
			LONG  MLastArchivedID;
			WORD  MLastUpdatedDate;
			WORD  MLastUpdatedTime;
			LONG  MLastUpdatedID;
			WORD  MLastAccessedDate;
			WORD  MInheritanceGrantMask;
			WORD  MInheritanceRevokeMask;
			int   MMaximumSpace;
			LONG  MLastUpdatedInSeconds;
		};
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		typedef struct
		{
			FLMUINT32	time_low;
			FLMUINT16	time_mid;
			FLMUINT16	time_hi_and_version;
			FLMBYTE		clk_seq_hi_res;
			FLMBYTE		clk_seq_low;
			FLMBYTE		node[6];
		} NWGUID;
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		typedef struct WSAData
		{
			  WORD					wVersion;
			  WORD					wHighVersion;
			  char					szDescription[ WSADESCRIPTION_LEN + 1];
			  char					szSystemStatus[ WSASYS_STATUS_LEN + 1];
			  unsigned short		iMaxSockets;
			  unsigned short		iMaxUdpDg;
			  char *					lpVendorInfo;
		} WSADATA, * LPWSADATA;
	
		/*************************************************************************
		Desc:
		*************************************************************************/
		struct in_addr
		{
			union 
			{
				struct 
				{ 
					unsigned char 		s_b1;
					unsigned char		s_b2;
					unsigned char 		s_b3;
					unsigned char		s_b4;
				} S_un_b;
				
				struct
				{
					unsigned short		s_w1;
					unsigned short		s_w2;
				} S_un_w;
				
				unsigned long S_addr;
			} S_un;
		};
			  
		/*************************************************************************
		Desc:
		*************************************************************************/
		struct sockaddr_in
		{
		  short   			sin_family;
		  unsigned short	sin_port;
		  struct in_addr 	sin_addr;
		  char    			sin_zero[8];
		};
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		struct sockaddr 
		{
			unsigned short	sa_family;
			char				sa_data[14];
		};
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		struct hostent 
		{
			char *			h_name;
			char **			h_aliases;
			short 			h_addrtype;
			short				h_length;
			char **			h_addr_list;
		};
			
		/*************************************************************************
		Desc:
		*************************************************************************/
		struct timeval 
		{
			long    tv_sec;
			long    tv_usec;
		};
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		typedef struct fd_set
		{
			unsigned int	fd_count;
			SOCKET  			fd_array[ FD_SETSIZE];
		} fd_set;
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		typedef struct _WSABUF 
		{
			unsigned long			len;
			char *					buf;
		} WSABUF, * LPWSABUF;
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		typedef struct
		{
			 ULONG		TokenRate;
			 ULONG		TokenBucketSize;
			 ULONG		PeakBandwidth;
			 ULONG		Latency;
			 ULONG		DelayVariation;
			 ULONG		ServiceType;
			 ULONG		MaxSduSize;
			 ULONG		MinimumPolicedSize;
		} FLOWSPEC, *PFLOWSPEC, * LPFLOWSPEC;
	
		/*************************************************************************
		Desc:
		*************************************************************************/
		typedef struct
		{
		 FLOWSPEC      SendingFlowspec;
		 FLOWSPEC      ReceivingFlowspec;
		 WSABUF        ProviderSpecific;
		} QOS, * LPQOS;
				
		extern "C" FLMBYTE F_NW_Default_Volume_Name[];
		
		extern "C" LONG FlaimToNWOpenFlags(
			FLMUINT			uiAccess,
			FLMBOOL			bDoDirectIo);
		
		extern "C" LONG GetCurrentClock(
			clockAndStatus *			dataPtr);
			
		extern "C" void GetSyncClockFields(
			LONG							bitMap, 
			Synchronized_Clock_T *	aClock);
				
		extern "C" void kYieldThread( void);
		
		extern "C" int kGetThreadName( 
			FLMUINT32 		ui32ThreadId,
			char *			szName, 
			int 				iBufSize);
		
		extern "C" void * kCreateThread(
			BYTE *			name,
			void * 			(*StartAddress)(void *, void *),
			void *			StackAddressHigh,
			LONG				StackSize,
			void *			Argument);
			
		extern "C" void kYieldIfTimeSliceUp( void);
		
		extern "C" int kSetThreadName(
			void *			ThreadHandle,
			BYTE *			buffer);
		
		extern "C" LONG kScheduleThread(
			void *			ThreadHandle);
		
		extern "C" ERROR kDelayThread( 
			UINT				uiMilliseconds);
		
		extern "C" void * kCurrentThread( void);
		
		extern "C" int kDestroyThread(
			void *			ThreadHandle);
		
		extern "C" void kExitThread(
			void *			ExitStatus);
		
		extern "C" LONG kSetThreadLoadHandle(
			void *			ThreadHandle,
			LONG				nlmHandle);
		
		extern "C" LONG GetRunningProcess( void);
		
		extern "C" void KillMe(
			LoadDefinitionStructure *	LoadRecord);
		
		extern "C" void NWYieldIfTime( void);
		
		extern "C" void CSetD( 
			LONG 				value, 
			void *			address,
			LONG 				numberOfDWords);
		
		extern "C" void CMovB( 
			void *			src, 
			void *			dst,
			LONG 				numberOfBytes);
		
		extern "C" void * Alloc( 
			LONG				numberOfBytes,
			rtag_t			lRTag);
			
		extern "C" void Free( 
			void *			address);
		
		extern "C" rtag_t AllocateResourceTag(
			void *			pvNLMHandle,
			const char *	pszDescription,
			uint32_t 		ResourceSignature);
		
		extern "C" LONG ReturnResourceTag(
			rtag_t			RTag,
			BYTE				displayErrorsFlag);
		
		extern "C" LONG ConvertPathString(
			LONG 				stationNumber,
			BYTE 				base,
			BYTE *			modifierString,
			LONG *			volumeNumber,
			LONG *			pathBase,
			BYTE *			pathString,
			LONG *			pathCount);
		
		extern "C" LONG GetEntryFromPathStringBase(
			LONG 				Station,
			LONG 				Volume,
			LONG 				PathBase,
			BYTE *			PathString,
			LONG 				PathCount,
			LONG 				SourceNameSpace,
			LONG 				DesiredNameSpace,
			struct DirectoryStructure **	Dir,
			LONG *			DirectoryNumber);
		
		extern "C" LONG NDSCreateStreamFile(
			LONG 				Station,
			LONG 				Task,
			BYTE *			fileName,
			LONG 				CreateAttributes,
			LONG *			fileHandle,
			LONG *			DOSDirectoryBase);
		
		extern "C" LONG NDSOpenStreamFile(
			LONG 				Station,
			LONG 				Task,
			BYTE *			fileName,
			LONG 				RequestedRights,
			LONG *			fileHandle,
			LONG *			DOSDirectoryBase);
		
		extern "C" LONG NDSDeleteStreamFile(
			LONG				Station,
			LONG				Task,
			BYTE *			fileName,
			LONG *			DOSDirectoryBase);
		
		extern "C" LONG EraseFile(
			LONG				Station,
			LONG 				Task,
			LONG 				Volume,
			LONG 				PathBase,
			BYTE *			PathString,
			LONG 				PathCount,
			LONG 				NameSpace,
			LONG 				MatchBits);
			
		extern "C" LONG RenameEntry(
			LONG				Station,
			LONG 				Task,
			LONG 				Volume,
			LONG 				PathBase,
			BYTE *			PathString,
			LONG 				PathCount,
			LONG 				NameSpace,
			LONG 				MatchBits,
			BYTE 				SubdirectoryFlag,
			LONG 				NewBase,
			BYTE *			NewString,
			LONG 				NewCount,
			LONG 				CompatabilityFlag,
			BYTE 				AllowRenamesToMyselfFlag);
			
		extern "C" LONG OpenFile(
			LONG 				Station,
			LONG 				Task,
			LONG 				Volume,
			LONG 				PathBase,
			BYTE *			PathString,
			LONG 				PathCount,
			LONG 				NameSpace,
			LONG 				MatchBits,
			LONG 				RequestedRights,
			BYTE 				DataStreamNumber,
			LONG *			Handle,
			LONG *			DirectoryNumber,
			void **			DirectoryEntry);
			
		extern "C" LONG CreateFile(
			LONG 				Station,
			LONG 				Task,
			LONG 				Volume,
			LONG 				PathBase,
			BYTE *			PathString,
			LONG 				PathCount,
			LONG 				NameSpace,
			LONG 				CreatedAttributes,
			LONG 				FlagBits,
			BYTE 				DataStreamNumber,
			LONG *			Handle,
			LONG *			DirectoryNumber,
			void **			DirectoryEntry);
	
		extern "C" LONG CloseFile(
			LONG 				station,
			LONG 				task,
			LONG 				handle);
			
		extern "C" LONG ReadFile(
			LONG 				stationNumber,
			LONG 				handle,
			LONG 				startingOffset,
			LONG 				bytesToRead,
			LONG *			actualBytesRead,
			void *			buffer);
		
		extern "C" LONG WriteFile(
			LONG 				stationNumber,
			LONG 				handle,
			LONG 				startingOffset,
			LONG 				bytesToWrite,
			void *			buffer);
		
		extern "C" LONG SetFileSize(
			LONG 				station,
			LONG				handle,
			LONG				filesize,
			LONG				truncateflag);
		
		extern "C" LONG GetFileSize(
			LONG 				stationNumber,
			LONG 				handle,
			LONG *			fileSize);
		
		extern "C" LONG DirectReadFile(
			LONG 				station,
			LONG 				handle,
			LONG 				startingsector,
			LONG 				sectorcount,
			BYTE *			buffer);
		
		extern "C" LONG SwitchToDirectFileMode(
			LONG 				station,
			LONG 				handle);
		
		extern "C" LONG ReturnVolumeMappingInformation(
			LONG 				volumenumber,
			struct VolumeInformationStructure * volumeInformation);
		
		extern "C" LONG ExpandFileInContiguousBlocks(
			LONG				station,
			LONG 				handle,
			LONG 				fileblocknumber,
			LONG 				numberofblocks,
			LONG 				vblocknumber,
			LONG 				segnumber);
		
		extern "C" LONG FreeLimboVolumeSpace(
			LONG				volumenumber,
			LONG				numberofblocks);
		
		extern "C" LONG DirectWriteFileNoWait(
			LONG				station,
			LONG 				handle,
			LONG 				startingsector,
			LONG 				sectorcount,
			BYTE *			buffer,
			void 				(*callbackroutine)(LONG, LONG, LONG),
			LONG 				callbackparameter);
		
		extern "C" LONG DirectWriteFile(
			LONG 				station,
			LONG 				handle,
			LONG 				startingsector,
			LONG 				sectorcount,
			BYTE *			buffer);
		
		extern "C" LONG RevokeFileHandleRights(
			LONG 				Station,
			LONG 				Task,
			LONG 				FileHandle,
			LONG 				QueryFlag,
			LONG 				removeRights,
			LONG *			newRights);
		
		extern "C" LONG ModifyDirectoryEntry(
			LONG				Station,
			LONG				Task,
			LONG				Volume,
			LONG				PathBase,
			BYTE *			PathString,
			LONG				PathCount,
			LONG				NameSpace,
			LONG				MatchBits,
			LONG				TargetNameSpace,
			struct ModifyStructure * ModifyVector,
			LONG 				ModifyBits,
			LONG 				AllowWildCardsFlag);
			
		extern "C" LONG MapFileHandleToFCB(
			LONG				handle,
			FCBType **		fcb);
	
		extern "C" LONG MapPathToDirectoryNumber(
			LONG				Station,
			LONG 				Volume,
			LONG 				PathBase,
			BYTE *			PathString,
			LONG 				PathCount,
			LONG 				NameSpace,
			LONG *			DirectoryNumber,
			LONG *			FileFlag);
				
		extern "C" LONG CreateDirectory(
			LONG				Station,
			LONG				Volume,
			LONG				PathBase,
			BYTE *			PathString,
			LONG				PathCount,
			LONG				NameSpace,
			LONG				DirectoryAccessMask,
			LONG *			ReturnedDirectoryNumber,
			void **			ReturnedSubDir);
			
		extern "C" LONG DeleteDirectory(
			LONG 				Station,
			LONG 				Volume,
			LONG 				PathBase,
			BYTE *			PathString,
			LONG 				PathCount,
			LONG 				NameSpace);
				
		extern "C" LONG DirectorySearch(
			LONG 				Station,
			LONG 				Volume,
			LONG 				DirectoryNumber,
			LONG 				NameSpace,
			LONG 				StartEntryNumber,
			BYTE *			Pattern,
			LONG 				MatchBits,
			struct DirectoryStructure **	DirectoryEntry,
			LONG *			ReturnedDirectoryNumber);
			
		extern "C" LONG VMGetDirectoryEntry(
			LONG				volumeNumber,
			LONG				directoryEntry,
			void *			directoryEntryPointer);
			
		extern "C" LONG ImportPublicSymbol(
			LONG				moduleHandle,
			BYTE *			symbolName);
	
		extern "C" LONG UnImportPublicSymbol(
			LONG				moduleHandle,
			BYTE *			symbolName);
	
		extern "C" LONG ExportPublicSymbol(
			LONG				moduleHandle,
			BYTE *			symbolName,
			LONG				address);
	
		extern "C" void SynchronizeStart( void);
	
		extern "C" void * CFindLoadModuleHandle( void *);
	
		extern "C" int atexit(
			F_EXIT_FUNC		fnExit);
			
		extern "C" LONG LoadModule(
			void *			screenID,
			BYTE *			fileName,
			LONG				loadOptions);
	
		extern "C" LONG UnloadModule( 
			void **			pScreenID, 
			const char *	commandline);
			
		extern "C" int SGUIDCreate( 
			NWGUID *			guidBfr);
	
		extern "C" void * GetSystemConsoleScreen( void);
	
		extern "C" LONG SizeOfAllocBlock( 
			void *			AllocAddress);
	
		extern "C" SEMAPHORE kSemaphoreAlloc(
			BYTE *			pSemaName,
			UINT				SemaCount);
	
		extern "C" ERROR kSemaphoreFree(
			SEMAPHORE		SemaHandle);
	
		extern "C" ERROR kSemaphoreWait(
			SEMAPHORE		SemaHandle);
	
		extern "C" ERROR kSemaphoreTimedWait(
			SEMAPHORE		SemaHandle, 
			UINT				MilliSecondTimeOut);
	
		extern "C" ERROR kSemaphoreSignal(
			SEMAPHORE		SemaHandle);
	
		extern "C" UINT kSemaphoreExamineCount(
			SEMAPHORE		SemaHandle);
	
		extern "C" MUTEX kMutexAlloc(
			BYTE *			MutexName);
	
		extern "C" ERROR kMutexFree(
			MUTEX				MutexHandle);
	
		extern "C" ERROR kMutexLock(
			MUTEX				MutexHandle);
	
		extern "C" ERROR kMutexUnlock(
			MUTEX				MutexHandle);
	
		extern "C" void CMoveFast( 
			const void *	src,
			void *			dst,
			LONG				numberOfBytes);
	
		extern "C" void EnterDebugger( void);
	
		extern "C" void GetClosestSymbol(
			BYTE *			szBuffer,
			LONG				udAddress);
	
		extern "C" LONG GetCurrentTime( void);
	
		extern "C" void ConvertTicksToSeconds(
			LONG				ticks,
			LONG *			seconds,
			LONG *			tenthsOfSeconds);
	
		extern "C" void ConvertSecondsToTicks(
			LONG				seconds,
			LONG				tenthsOfSeconds,
			LONG *			ticks);
	
		extern "C" LONG GetCacheBufferSize(void);
	
		extern "C" LONG GetOriginalNumberOfCacheBuffers(void);
	
		extern "C" LONG GetCurrentNumberOfCacheBuffers(void);
	
		extern "C" LONG GetNLMAllocMemoryCounts(
			void *			moduleHandle,
			FLMUINT *		freeBytes,
			FLMUINT *		freeNodes,
			FLMUINT *		allocatedBytes,
			FLMUINT *		allocatedNodes,
			FLMUINT *		totalMemory);
	
		extern "C" LONG atomic_xchg( 
			volatile LONG * 	address,
			LONG 				 	value);
	
		extern "C" FLMINT32 nlm_AtomicIncrement( 
			volatile LONG *	piTarget);
	
		extern "C" FLMINT32 nlm_AtomicDecrement( 
			volatile LONG *	piTarget);
	
		#define nlm_AtomicExchange( piTarget, iValue) \
			((FLMINT32)atomic_xchg( (volatile LONG *)(piTarget), (LONG)(iValue)))
	
		/*************************************************************************
		Desc:
		*************************************************************************/
		#if !defined( __MWERKS__)
			#pragma aux nlm_AtomicIncrement parm [ecx];
			#pragma aux nlm_AtomicIncrement = \
			0xB8 0x01 0x00 0x00 0x00   		/*  mov	eax, 1  	 			*/ \
			0xF0 0x0F 0xC1 0x01					/*  lock xadd [ecx], eax 	*/ \
			0x40										/*  inc	eax 					*/ \
			parm [ecx]	\
			modify exact [eax];
	
			#pragma aux nlm_AtomicDecrement parm [ecx];
			#pragma aux nlm_AtomicDecrement = \
			0xB8 0xFF 0xFF 0xFF 0xFF   		/*  mov	eax, 0ffffffffh	*/ \
			0xF0 0x0F 0xC1 0x01					/*  lock xadd [ecx], eax 	*/ \
			0x48										/*  dec	eax 					*/ \
			parm [ecx]	\
			modify exact [eax];
		#endif	
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		#if defined( __MWERKS__)
		FINLINE FLMINT32 nlm_AtomicIncrement(
			volatile LONG *	piTarget)
		{
			FLMINT32				i32Result;
	
			__asm
			{
				mov	eax, 1
				mov	ecx, piTarget
				lock xadd [ecx], eax
				inc	eax
				mov	i32Result, eax
			}
	
			return( i32Result);
		}
		#endif
	
		/*************************************************************************
		Desc:
		*************************************************************************/
		#if defined( __MWERKS__)
		FINLINE FLMINT32 nlm_AtomicDecrement(
			volatile LONG *	piTarget)
		{
			FLMINT32				i32Result;
	
			__asm
			{
				mov	eax, 0ffffffffh
				mov	ecx, piTarget
				lock xadd [ecx], eax
				dec	eax
				mov	i32Result, eax
			}
	
			return( i32Result);
		}
		#endif
			
		/*************************************************************************
		Desc:
		*************************************************************************/
		#define FD_CLR(fd, set) do { \
			 unsigned int __i; \
			 for (__i = 0; __i < ((fd_set *)(set))->fd_count ; __i++) { \
				  if (((fd_set *)(set))->fd_array[__i] == fd) { \
						while (__i < ((fd_set *)(set))->fd_count-1) { \
							 ((fd_set *)(set))->fd_array[__i] = \
								  ((fd_set *)(set))->fd_array[__i+1]; \
							 __i++; \
						} \
						((fd_set *)(set))->fd_count--; \
						break; \
				  } \
			 } \
		} while(0)
		
		/*************************************************************************
		Desc:
		*************************************************************************/
		#define FD_SET(fd, set) do { \
			 unsigned int __i; \
			 for (__i = 0; __i < ((fd_set *)(set))->fd_count; __i++) { \
				  if (((fd_set *)(set))->fd_array[__i] == (fd)) { \
						break; \
				  } \
			 } \
			 if (__i == ((fd_set *)(set))->fd_count) { \
				  if (((fd_set *)(set))->fd_count < FD_SETSIZE) { \
						((fd_set *)(set))->fd_array[__i] = (fd); \
						((fd_set *)(set))->fd_count++; \
				  } \
			 } \
		} while(0)
	
		#define FD_ZERO(set) \
			(((fd_set *)(set))->fd_count=0)
	
		#define FD_ISSET(fd, set) \
			__WSAFDIsSet((SOCKET)(fd), (fd_set *)(set))
	
		#define MAKEWORD(low,high) \
				  ((WORD)(((BYTE)(low)) | ((WORD)((BYTE)(high))) << 8))
		
		extern "C" int WSAStartup(
			unsigned short	wVersionRequested,
			LPWSADATA 		lpWSAData);
	
		extern "C" int WSACleanup( void);
	
		extern "C" int gethostname(
			char *			name,
			int				namelen);
			
		extern "C" struct hostent * gethostbyname(
			const char *	name);
		 
		extern "C" struct hostent * gethostbyaddr(
			const char *	addr,
			int				len,
			int				type);
		 
		extern "C" char * inet_ntoa(
			struct in_addr in);
			
		extern "C" int select(
			int				nfds,
			fd_set * 		readfds,
			fd_set * 		writefds,
			fd_set *			exceptfds,
			const struct timeval * timeout);
	
		extern "C" int __WSAFDIsSet( SOCKET, fd_set *);
		
		extern "C" int recv(
			SOCKET			s,
			char *			buf,
			int				len,
			int				flags);
		 
		extern "C" int send(
			SOCKET 			s,
			const char *	buf,
			int 				len,
			int 				flags);
			
		extern "C" int setsockopt(
			SOCKET			s,
			int				level,
			int				optname,
			const char *	optval,
			int 				optlen);
			
		extern "C" int closesocket(
			SOCKET			s);
	
		extern "C" SOCKET socket(
			int 				af,
			int 				type,
			int 				protocol);
			
		extern "C" int bind(
			SOCKET 			s,
			const struct sockaddr * name,
			int 				namelen);
			
		extern "C" int listen(
			SOCKET			s,
			int				backlog);
	
		extern "C" unsigned long inet_addr(
			const char * 	cp);
			
		extern "C" unsigned short htons(
			unsigned short	hostshort);
	
		extern "C" unsigned long htonl(
			unsigned long hostlong);
			
		extern "C" int WSAGetLastError( void);
		
		extern "C" int WSAConnect(
			SOCKET 			s,
			const struct sockaddr * name,
			int 				namelen,
			LPWSABUF 		lpCallerData,
			LPWSABUF 		lpCalleeData,
			LPQOS 			lpSQOS,
			LPQOS 			lpGQOS);
			
		extern "C" SOCKET WSAAccept(
			SOCKET 				s,
			struct sockaddr * addr,
			int *					addrlen,
			void * 				lpfnCondition,
			DWORD 				dwCallbackData);
			
		extern "C" int OpenScreen( 
			const char *		name,
			rtag_t				rTag,
			scr_t *				newScrID);
			
		extern "C" void ActivateScreen( 
			scr_t					scrID);
	
		extern "C" void CloseScreen( 
			scr_t					scrID);
			
		extern "C" int DisplayScreenTextWithAttribute( 
			scr_t					scrID,
			uint32_t				line,
			uint32_t				col,
			uint32_t				length,
			uint8_t				lineAttr,
			char *				text);
			
		extern "C" void ClearScreen(
			scr_t					scrID);
		
		extern "C" void GetScreenSize(
			uint16_t *			height,
			uint16_t *			width);
	
		extern "C" void DisableInputCursor(
			scr_t					scrID);
	
		extern "C" void EnableInputCursor( 
			scr_t					scrID);
	
		extern "C" void SetCursorStyle( 
			scr_t					scrID,
			uint16_t				newStyle);
	
		extern "C" int PositionOutputCursor( 
			scr_t					scrID,
			uint16_t				row,
			uint16_t				col);
	
		extern "C" int UngetKey( 
			scr_t					scrID,
			uint8_t				type,
			uint8_t				value,
			uint8_t				status,
			uint8_t				scancode);
				
		extern "C" void GetKey( 
			scr_t					scrID,
			uint8_t *			type,
			uint8_t *			value,
			uint8_t *			status,
			uint8_t *			scancode,
			size_t				linesToProtect);
				
		extern "C" void PositionInputCursor(
			scr_t					scrID,
			uint16_t				row,
			uint16_t				col);
	
		extern "C" int CheckKeyStatus(
			scr_t					scrID);
		
		RCODE f_netwareStartup( void);
			
		void f_netwareShutdown( void);
		
	#endif // FLM_RING_ZERO_NLM

#pragma pack(pop)

#endif // FLM_NLM
#endif // FTKNLM_H
