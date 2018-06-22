//------------------------------------------------------------------------------
// Desc:	This file contains the routine which calculates a block checksum.
// Tabs:	3
//
// Copyright (c) 1999-2007 Novell, Inc. All Rights Reserved.
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

#include "flaimsys.h"

#if (defined( FLM_WIN) && !defined( FLM_64BIT)) || defined( FLM_NLM)

	static unsigned long gv_mmxCheckSumFlag = 1;
	
	#if defined( FLM_WATCOM_NLM)
	
		extern void FastBlockCheckSumMMX(
				void *			pBlk,
				unsigned long *puiChecksum,	
				unsigned long *puiXORdata,
				unsigned long	uiNumberOfBytes);
		
		extern void FastBlockCheckSum386(
				void *			pBlk,
				unsigned long *puiChecksum,	
				unsigned long *puiXORdata,
				unsigned long	uiNumberOfBytes);
		
		extern unsigned long GetMMXSupported(void);
		
	#else
	
		static void FastBlockCheckSumMMX(
				void *			pBlk,
				unsigned long *puiChecksum,	
				unsigned long *puiXORdata,
				unsigned long	uiNumberOfBytes);
		
		static void FastBlockCheckSum386(
				void *			pBlk,
				unsigned long *puiChecksum,	
				unsigned long *puiXORdata,
				unsigned long	uiNumberOfBytes);
		
		static unsigned long GetMMXSupported(void);
		
	#endif
	
#endif

/********************************************************************
Desc: Returns 1 if the CPU supports MMX
Ret:	0 or 1 if CPU supports MMX
*********************************************************************/
#if defined( FLM_WATCOM_NLM)

	#pragma aux GetMMXSupported parm;
	#pragma aux GetMMXSupported = \
		0xB8 0x01 0x00 0x00 0x00            /* mov		eax, 1  	*/\
		0x0F 0xA2                         	/* CPUID  				*/\
		0x33 0xC0                         	/* xor		eax, eax 	*/\
		0xF7 0xC2 0x00 0x00 0x80 0x00       /* test		edx, (1 SHL 23) ;check for MMX Instruction support  */\
		0x0F 0x95 0xC0                      /* setnz	al  		*/\
		modify exact [EAX EBX ECX EDX];

#elif defined( FLM_WIN) && !defined( FLM_64BIT)

	unsigned long GetMMXSupported( void)
	{
		unsigned long bMMXSupported;
		__asm
		{
			mov		eax, 1
			cpuid
			xor		eax, eax
			test		edx, (1 SHL 23)
			setnz		al
			mov		bMMXSupported, eax
		}
		return bMMXSupported;
	}
	
#endif

/********************************************************************
Desc: Performs part of the FLAIM block checksum algorithm 
		using MMX instructions.
*********************************************************************/
#if defined( FLM_WATCOM_NLM)

	#pragma aux FastBlockCheckSumMMX parm [ESI] [eax] [ebx] [ecx];
	#pragma aux FastBlockCheckSumMMX = \
		0x50                          /* push	eax			;save the sum pointer  								*/\
		0x53                          /* push	ebx			;save the xor pointer 								*/\
		0x8B 0x10                     /* mov	edx, [eax]	;for local add 										*/\
		0x81 0xE2 0xFF 0x00 0x00 0x00 /* and	edx, 0ffh	;clear unneeded bits									*/\
		0x8B 0x1B                     /* mov	ebx, [ebx]	;for local xor 										*/\
		0x81 0xE3 0xFF 0x00 0x00 0x00 /* and	ebx, 0ffh	;clear unneeded bits									*/\
		0x8B 0xF9                     /* mov	edi, ecx		;save the amount to copy							*/\
		0x83 0xF9 0x20                /* cmp	ecx, 32		;see if we have enough for the big loop		*/\
		0x0F 0x82 0x63 0x00 0x00 0x00	/* jb		#MediumStuff 															*/\
		0xC1 0xE9 0x05                /* shr	ecx, 5		;convert length to 32 byte blocks				*/\
		0x83 0xE7 0x1F						/* and	edi, 01fh	;change saved length to remainder				*/\
		0x0F 0xEF 0xED						/* pxor	mm5, mm5		;wasted space to 16 byte align the loop		*/\
		0x0F 0x6E 0xE2						/* movd	mm4,edx																	*/\
		0x0F 0x6E 0xEB						/* movd	mm5,ebx																	*/\
		0x0F 0x6F 0x06						/* movq	mm0, [esi] 																*/\
		0x0F 0x6F 0x4E 0x08				/* movq	mm1, [esi + 8] 														*/\
		0x0F 0x6F 0x56 0x10				/* movq	mm2, [esi + 16] 														*/\
		0x0F 0x6F 0x5E 0x18				/* movq	mm3, [esi + 24] 														*/\
		0x83 0xC6 0x20                /* add	esi, 32	;move the data pointer ahead 32						*/\
		0x0F 0xFC 0xE0						/* paddb	mm4, mm0 																*/\
		0x0F 0xEF 0xE8						/* pxor	mm5, mm0 																*/\
		0x0F 0xFC 0xE1						/* paddb	mm4, mm1 																*/\
		0x0F 0xEF 0xE9						/* pxor	mm5, mm1 																*/\
		0x0F 0xFC 0xE2						/* paddb	mm4, mm2 																*/\
		0x0F 0xEF 0xEA						/* pxor	mm5, mm2 																*/\
		0x0F 0xFC 0xE3						/* paddb	mm4, mm3 																*/\
		0x0F 0xEF 0xEB						/* pxor	mm5, mm3 																*/\
		0x49                          /* dec	ecx		;see if there is more to do							*/\
		0x75 0xD3                     /* jnz	#BigStuffLoop 															*/\
		0x0F 0x7E 0xEB						/* movd	ebx, mm5 																*/\
		0x0F 0x73 0xD5 0x20				/* psrlq	mm5, 32 																	*/\
		0x0F 0x7E 0xE8						/* movd	eax, mm5 																*/\
		0x33 0xD8                     /* xor	ebx, eax 																*/\
		0x0F 0x6F 0xC4						/* movq	mm0, mm4 																*/\
		0x0F 0x73 0xD0 0x20				/* psrlq	mm0, 32 																	*/\
		0x0F 0xFC 0xE0						/* paddb	mm4, mm0 																*/\
		0x0F 0x6F 0xC4						/* movq	mm0, mm4 																*/\
		0x0F 0x73 0xD0 0x10				/* psrlq	mm0, 16 																	*/\
		0x0F 0xFC 0xE0						/* paddb	mm4, mm0 																*/\
		0x0F 0x7E 0xE2						/* movd	edx, mm4 																*/\
		0x0F 0x77							/* emms	;end of MMX stuff 													*/\
		0x8B 0xCF                     /* mov	ecx, edi	;load up the rest of the length						*/\
		0x83 0xF9 0x04                /* cmp	ecx, 4 																	*/\
		0x0F 0x82 0x1D 0x00 0x00 0x00	/* jb		#SmallStuff 															*/\
		0xC1 0xE9 0x02                /* shr	ecx, 2 																	*/\
		0x83 0xE7 0x03                /* and	edi, 3 																	*/\
		0x8B 0x06                     /* mov	eax, [esi] 																*/\
		0x83 0xC6 0x04                /* add	esi, 4 																	*/\
		0x33 0xD8                     /* xor	ebx, eax 																*/\
		0x02 0xD0                     /* add	dl, al 																	*/\
		0x02 0xF4                     /* add	dh, ah 																	*/\
		0xC1 0xE8 0x10                /* shr	eax, 16 																	*/\
		0x02 0xD0                     /* add	dl, al 																	*/\
		0x02 0xF4                     /* add	dh, ah 																	*/\
		0x49                          /* dec	ecx 																		*/\
		0x75 0xEB                     /* jnz	#DSSumLoop 																*/\
		0x8B 0xCF                     /* mov	ecx, edi	;load up the rest of the length						*/\
		0x02 0xD6                     /* add	dl, dh	;get complete sum in dl									*/\
		0x8B 0xC3                     /* mov	eax, ebx	;get complete xor in bl									*/\
		0xC1 0xE8 0x10                /* shr	eax, 16 																	*/\
		0x66 0x33 0xD8                /* xor	bx, ax 																	*/\
		0x32 0xDF                     /* xor	bl, bh 																	*/\
		0x83 0xF9 0x00                /* cmp	ecx, 0	;see if anything left to do - 3 or less bytes	*/\
		0x0F 0x84 0x0A 0x00 0x00 0x00	/* jz		#Done 																	*/\
		0x8A 0x06                     /* mov	al, [esi] 																*/\
		0x46                          /* inc	esi																		*/\
		0x02 0xD0                     /* add	dl, al 																	*/\
		0x32 0xD8                     /* xor	bl, al 																	*/\
		0x49                          /* dec	ecx 																		*/\
		0x75 0xF6                     /* jnz	#SmallStuffLoop 														*/\
		0x81 0xE2 0xFF 0x00 0x00 0x00 /* and	edx, 0ffh	;clear unneeded bits									*/\
		0x58                          /* pop	eax 																		*/\
		0x81 0xE3 0xFF 0x00 0x00 0x00 /* and	ebx, 0ffh	;clear unneeded bits									*/\
		0x5F                          /* pop	edi 																		*/\
		0x89 0x18                     /* mov	[eax], ebx 																*/\
		0x89 0x17                     /* mov	[edi], edx 																*/\
		parm [ESI] [eax] [ebx] [ecx]	\
		modify exact [eax ebx ecx edx ESI EDI];

#elif defined( FLM_WIN) && !defined( FLM_64BIT)

	static void FastBlockCheckSumMMX(
			void *				pBlk,
			unsigned long *	puiChecksum,	
			unsigned long *	puiXORdata,
			unsigned long		uiNumberOfBytes)
	{
		__asm
		{
				mov		esi, pBlk
	
				// Load up the starting checksum values into edx (add) and ebx (XOR)
	
				mov		eax, puiChecksum
				mov		edx, [eax]
				and		edx, 0ffh			;clear unneeded bits 
				mov		eax, puiXORdata
				mov		ebx, [eax]
				and		ebx, 0ffh			;clear unneeded bits 
				mov		ecx, uiNumberOfBytes
				mov		edi, ecx				;save the amount to copy 
	
				cmp		ecx, 32				;see if we have enough for the big loop 
				jb			MediumStuff 					
	
				shr		ecx, 5				;convert length to 32 byte blocks
				and		edi, 01fh			;change saved length to remainder
				pxor		mm5, mm5				;wasted space to 16 byte align the upcoming loop - check tHIS..
				
				movd		mm4, edx				;set ADD
				movd		mm5, ebx				;set XOR
	
	BigStuffLoop:
													;load up mm0 - mm3 with 8 bytes each of data.
				movq		mm0, [esi]
				movq		mm1, [esi + 8]
				movq		mm2, [esi + 16]
				movq		mm3, [esi + 24]
				add		esi, 32				;move the data pointer ahead 32
													;add mm0 - mm3 to mm4
													;xor mm0 - mm3 with mm5
				paddb		mm4, mm0
				pxor		mm5, mm0
				paddb		mm4, mm1
				pxor		mm5, mm1
				paddb		mm4, mm2
				pxor		mm5, mm2
				paddb		mm4, mm3
				pxor		mm5, mm3
				dec		ecx					;see if there is more to do
				jnz		BigStuffLoop 
													;mm4 contains the sum to this point
													;mm5 contains the xor to this point
													;edi contains the bytes left 
													;esi points to data left to do
													;extract the xor value from mm5 and put it in ebx
				movd		ebx, mm5
				psrlq		mm5, 32 
				movd		eax, mm5
				xor		ebx, eax
													;extract the sum value from mm4 and put it in dl & dh 
				movq		mm0, mm4
				psrlq		mm0, 32 
				paddb		mm4, mm0
				movq		mm0, mm4
				psrlq		mm0, 16 
				paddb		mm4, mm0
				movd		edx, mm4
				emms								;end of MMX stuff
	
				mov		ecx, edi				;load up the rest of the length 
													;dl contains half the sum to this point
													;dh contains half the sum to this point
													;ebx contains the xor to this point - 32 bits wide.
													;ecx contains the bytes still left to do 
													;esi contains pointer to data to checksum 
	MediumStuff:
				cmp		ecx, 4
				jb			SmallStuff
				shr		ecx, 2
				and		edi, 3
	
	DSSumLoop:
				mov		eax, [esi]
				add		esi, 4
				xor		ebx, eax
				add		dl, al
				add		dh, ah
				shr		eax, 16
				add		dl, al
				add		dh, ah
				dec		ecx
				jnz		DSSumLoop
				mov		ecx, edi				;load up the rest of the length 
													;dl contains half the sum to this point 
													;dh contains half the sum to this point 
													;ebx contains the xor to this point - 32 bits wide.
													;ecx contains the bytes still left to do 
													;esi contains pointer to data to checksum 
	SmallStuff:
				add		dl, dh				;get complete sum in dl 
				mov		eax, ebx				;get complete xor in bl
				shr		eax, 16 						
				xor		bx, ax 							
				xor		bl, bh 							
				cmp		ecx, 0				;see if anything left to do - 3 or less bytes 
				jz			Done 							
	
	SmallStuffLoop: 						
				mov		al, [esi] 						
				inc		esi								
				add		dl, al 							
				xor		bl, al 							
				dec		ecx 							
				jnz		SmallStuffLoop 				
	Done: 									
				and		edx, 0ffh			;clear unneeded bits 
				and		ebx, 0ffh			;clear unneeded bits 
	
				// Set the return values.
	
				mov		eax, puiChecksum
				mov		[eax], edx
	
				mov		eax, puiXORdata
				mov		[eax], ebx
		}
		return;
	}
#endif

/******************************************************************************
Desc: Performs part of the FLAIM block checksum algorithm 
		using 386 and NOT MMX instructions.
******************************************************************************/
#if defined( FLM_WATCOM_NLM)

	#pragma aux FastBlockCheckSum386 parm [ESI] [eax] [ebx] [ecx];

	#pragma aux FastBlockCheckSum386 = \
	0x50                            /* push	eax			;save the sum pointer  	*/\
	0x53                            /* push	ebx			;save the xor pointer 	*/\
	0x8B 0x10                       /* mov		edx, [eax]	;for local add 			*/\
	0x81 0xE2 0xFF 0x00 0x00 0x00	  /* and		edx, 0ffh	;clear unneeded bits		*/\
	0x8B 0x1B                       /* mov		ebx, [ebx]	;for local xor 			*/\
	0x81 0xE3 0xFF 0x00 0x00 0x00   /* and		ebx, 0ffh	;clear unneeded bits		*/\
											  /* ;dl contains the sum to this point 			*/\
											  /* ;ebx contains the xor to this point			*/\
											  /* ;ecx contains the bytes still left to do	*/\
											  /* ;esi contains pointer to data to checksum	*/\
	0x83 0xF9 0x04                  /* cmp		ecx, 4 										*/\
	0x0F 0x82 0x1F 0x00 0x00 0x00	  /* jb		#SmallStuff 								*/\
	0x8B 0xF9                       /* mov		edi, ecx 									*/\
	0xC1 0xE9 0x02                  /* shr		ecx, 2 										*/\
	0x83 0xE7 0x03                  /* and		edi, 3 										*/\
											  /* #DSSumLoop: 											*/\
	0x8B 0x06                       /* mov		eax, [esi] 									*/\
	0x83 0xC6 0x04                  /* add		esi, 4 										*/\
	0x33 0xD8                       /* xor		ebx, eax 									*/\
	0x02 0xD0                       /* add		dl, al 										*/\
	0x02 0xF4                       /* add		dh, ah 										*/\
	0xC1 0xE8 0x10                  /* shr		eax, 16 										*/\
	0x02 0xD0                       /* add		dl, al 										*/\
	0x02 0xF4                       /* add		dh, ah 										*/\
	0x49                            /* dec		ecx 											*/\
	0x75 0xEB                       /* jnz		#DSSumLoop 									*/\
	0x8B 0xCF                       /* mov		ecx, edi	;load up the rest of len	*/\
											  /* ;dl contains half the sum to this point 	*/\
											  /* ;dh contains half the sum to this point 	*/\
											  /* ;ebx contains the xor to this point			*/\
											  /* ;ecx contains the bytes still left to do	*/\
											  /* ;esi contains pointer to data to checksum	*/\
											  /* #SmallStuff: 										*/\
	0x02 0xD6                       /* add		dl, dh		;get complete sum in dl */\
	0x8B 0xC3                       /* mov		eax, ebx	;get complete xor in bl		*/\
	0xC1 0xE8 0x10                  /* shr		eax, 16 										*/\
	0x66 0x33 0xD8                  /* xor		bx, ax 										*/\
	0x32 0xDF                       /* xor		bl, bh 										*/\
	0x83 0xF9 0x00                  /* cmp		ecx, 0										*/\
	0x0F 0x84 0x0A 0x00 0x00 0x00	  /* jz		#Done 										*/\
											  /* #SmallStuffLoop: 									*/\
	0x8A 0x06                       /* mov		al, [esi] 									*/\
	0x46                            /* inc		esi											*/\
	0x02 0xD0                       /* add		dl, al 										*/\
	0x32 0xD8                       /* xor		bl, al 										*/\
	0x49                            /* dec		ecx 											*/\
	0x75 0xF6                       /* jnz		#SmallStuffLoop 							*/\
											  /* #Done: 												*/\
	0x81 0xE2 0xFF 0x00 0x00 0x00   /* and		edx, 0ffh	;clear unneeded bits		*/\
	0x58                            /* pop		eax 											*/\
	0x81 0xE3 0xFF 0x00 0x00 0x00   /* and		ebx, 0ffh	;clear unneeded bits		*/\
	0x5F                            /* pop		edi 											*/\
	0x89 0x18                       /* mov		[eax], ebx 									*/\
	0x89 0x17                       /* mov		[edi], edx 									*/\
	parm [ESI] [eax] [ebx] [ecx]	\
	modify exact [eax ebx ecx edx ESI EDI];

#elif defined( FLM_WIN) && !defined( FLM_64BIT)

	static void FastBlockCheckSum386(
			void *			pBlk,
			unsigned long *puiChecksum,	
			unsigned long *puiXORdata,
			unsigned long	uiNumberOfBytes)
	{
		__asm
		{
				mov		esi, pBlk

				// Load up the starting checksum values into edx (add) and ebx (XOR)

				mov		eax, puiChecksum
				mov		edx, [eax]				// Set local add
				and		edx, 0ffh			;clear unneeded bits 
				mov		eax, puiXORdata
				mov		ebx, [eax]
				and		ebx, 0ffh			;clear unneeded bits 
				mov		ecx, uiNumberOfBytes

											;dl contains the sum to this point 		
											;ebx contains the xor to this point - 32 bits wide. 
											;ecx contains the bytes still left to do 
											;esi contains pointer to data to checksum 
				cmp		ecx, 4 							
				jb			SmallStuff 					
				mov		edi, ecx 						
				shr		ecx, 2 							
				and		edi, 3

	DSSumLoop: 								
				mov		eax, [esi] 						
				add		esi, 4 							
				xor		ebx, eax 						
				add		dl, al 							
				add		dh, ah 							
				shr		eax, 16 						
				add		dl, al 							
				add		dh, ah 							
				dec		ecx 							
				jnz		DSSumLoop 						
				mov		ecx, edi		;load up the rest of the length 
											;dl contains half the sum to this point 	
											;dh contains half the sum to this point 	
											;ebx contains the xor to this point - 32 bits wide. 
											;ecx contains the bytes still left to do 
											;esi contains pointer to data to checksum

	SmallStuff: 							
				add		dl, dh		;get complete sum in dl 
				mov		eax, ebx		;get complete xor in bl
				shr		eax, 16 						
				xor		bx, ax 							
				xor		bl, bh 							
				cmp		ecx, 0		;see if anything left to do - 3 or less bytes 
				jz			Done 							

	SmallStuffLoop: 						
				mov		al, [esi] 						
				inc		esi								
				add		dl, al 							
				xor		bl, al 							
				dec		ecx 							
				jnz		SmallStuffLoop

	Done:
				and		edx, 0ffh	;clear unneeded bits 
				and		ebx, 0ffh	;clear unneeded bits 
			
				// Set the return values.

				mov		eax, puiChecksum		// Address of add result/start
				mov		[eax], edx

				mov		eax, puiXORdata		// Address of xor result/start
				mov		[eax], ebx
		}

		return;
	}
#endif

/******************************************************************************
Desc: Performs part of the FLAIM block checksum algorithm 
		using MMX or 386 instructions.
Note:	FastBlockCheckSum will start with the checksum and xordata you
		pass in.  It assumes that the data is already dword aligned.
******************************************************************************/
#if (defined( FLM_WIN) && !defined( FLM_64BIT)) || defined( FLM_NLM)
void FastBlockCheckSum(
		void *			pBlk,
		FLMUINT *		puiChecksum,	
		FLMUINT *		puiXORdata,
		FLMUINT			uiNumberOfBytes)
{
	if( gv_mmxCheckSumFlag == 1)
	{
		FastBlockCheckSumMMX( (void *) pBlk, (unsigned long *) puiChecksum, 
					(unsigned long *) puiXORdata, (unsigned long) uiNumberOfBytes);
	}
	else
	{
		FastBlockCheckSum386( (void *) pBlk, (unsigned long *) puiChecksum, 
					(unsigned long *) puiXORdata, (unsigned long) uiNumberOfBytes);
	}
}
#endif

/******************************************************************************
Desc: Sets the global variable to check if MMX instructions are allowed.
******************************************************************************/
void F_DbSystem::initFastBlockCheckSum( void)
{
#if (defined( FLM_WIN) && !defined( FLM_64BIT)) || defined( FLM_NLM)
	// NOTE that GetMMXSupported assumes that we are running on at least a
	// pentium.  The check to see if we are on a pentium requires that  we
	// modify the flags register, and we can't do that if we are running
	// in ring3.  Because NetWare 5 - according to our product marketing -
	// requires at least a P5 90Mhz, we will be safe.  When you port this
	// code to NT, you may need to come up with a safe way to see if we
	// can do MMX instructions - unless you can assume that even on NT you
	// will be on at least a P5.

	gv_mmxCheckSumFlag = GetMMXSupported();
#endif
}

/********************************************************************
Desc: Calculate the checksum for a packet.
*********************************************************************/
FLMBYTE RflCalcChecksum(
	FLMBYTE *	pucPacket,
	FLMUINT		uiPacketBodyLen)
{
	FLMUINT		uiBytesToChecksum;
	FLMUINT		uiChecksum;
	FLMBYTE *	pucStart;
	
	// Checksum is calculated for every byte in the packet that
	// comes after the checksum byte.

	uiBytesToChecksum = (FLMUINT)(uiPacketBodyLen +
									RFL_PACKET_OVERHEAD -
									(RFL_PACKET_CHECKSUM_OFFSET + 1));
									
	pucStart = &pucPacket [RFL_PACKET_CHECKSUM_OFFSET + 1];

	FLMBYTE *	pucEnd;
	FLMBYTE *	pucSectionEnd;
	FLMBYTE *	pucCur;
	FLMBYTE		ucTmp;
	
	uiChecksum = 0;
	pucCur = pucStart;
	pucEnd = pucStart + uiBytesToChecksum;

#ifdef FLM_64BIT
	pucSectionEnd = pucStart + (sizeof( FLMUINT) - ((FLMUINT)pucStart & 0x7));
#else
	pucSectionEnd = pucStart + (sizeof( FLMUINT) - ((FLMUINT)pucStart & 0x3));
#endif

	if( pucSectionEnd > pucEnd)
	{
		pucSectionEnd = pucEnd;
	}

	while( pucCur < pucSectionEnd)
	{
		uiChecksum = (uiChecksum << 8) + *pucCur++;
	}

#ifdef FLM_64BIT
	pucSectionEnd = (FLMBYTE *)((FLMUINT)pucEnd & 0xFFFFFFFFFFFFFFF8); 
#else
	pucSectionEnd = (FLMBYTE *)((FLMUINT)pucEnd & 0xFFFFFFFC); 
#endif

	while( pucCur < pucSectionEnd)
	{
		uiChecksum ^= *((FLMUINT *)pucCur);
		pucCur += sizeof( FLMUINT);
	}

	while( pucCur < pucEnd)
	{
		uiChecksum ^= *pucCur++;
	}

	ucTmp = (FLMBYTE)uiChecksum;

	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE)uiChecksum;

	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE)uiChecksum;

#ifdef FLM_64BIT
	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE)uiChecksum;

	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE)uiChecksum;

	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE)uiChecksum;

	uiChecksum >>= 8;
	ucTmp ^= (FLMBYTE)uiChecksum;
#endif

	ucTmp ^= (FLMBYTE)(uiChecksum >> 8);

	if( (uiChecksum = (FLMUINT)ucTmp) == 0)
	{
		uiChecksum = 1;
	}

	return( (FLMBYTE)uiChecksum);
}

/********************************************************************
Desc:	Calculate the checksum for a block.  NOTE: This is ALWAYS done
		on the raw image that will be written to disk.  This means
		that if the block needs to be converted before writing it out,
		it should be done before calculating the checksum.
*********************************************************************/
FLMUINT32 calcBlkCRC(
	F_BLK_HDR *		pBlkHdr,
	FLMUINT			uiBlkEnd)
{
	FLMUINT32	ui32SaveCRC;
	FLMUINT		uiAdds;
	FLMUINT		uiXORs;
	FLMBYTE *	pucBlkPtr;

	// Calculate CRC on everything except for the ui32BlkCRC value.
	// To do this, we temporarily change it to zero.  The saved
	// value will be restored after calculating the CRC.

	ui32SaveCRC = pBlkHdr->ui32BlkCRC;
	pBlkHdr->ui32BlkCRC = 0;
	uiAdds = 0;
	uiXORs = 0;
	pucBlkPtr = (FLMBYTE *)pBlkHdr;

#if defined( FLM_NLM) || (defined( FLM_WIN) && !defined( FLM_64BIT))

	FastBlockCheckSum( pucBlkPtr, &uiAdds, &uiXORs,
		(unsigned long)uiBlkEnd);

#else
	
	FLMBYTE *		pucCur = pucBlkPtr;
	FLMBYTE *		pucEnd = pucCur + uiBlkEnd;

	while( pucCur < pucEnd)	
	{
		uiAdds += *pucCur;
		uiXORs ^= *pucCur++;
	}

	uiAdds &= 0xFF;
#endif

	// Restore the CRC that was in the block.

	pBlkHdr->ui32BlkCRC = ui32SaveCRC;
	return( (FLMUINT32)((uiAdds << 16) + uiXORs));
}

/********************************************************************
Desc:	Calculate the CRC for the database header.
*********************************************************************/
FLMUINT32 calcDbHdrCRC(
	SFLM_DB_HDR *	pDbHdr)
{
	FLMUINT32	ui32SaveCRC;
	FLMUINT		uiAdds;
	FLMUINT		uiXORs;
	FLMBYTE *	pucHdr;

	// Checksum everything except for the ui32HdrCRC value.

	ui32SaveCRC = pDbHdr->ui32HdrCRC;
	pDbHdr->ui32HdrCRC = 0;

	uiAdds = 0; 
	uiXORs = 0;
	pucHdr = (FLMBYTE *)pDbHdr;

#if defined( FLM_NLM) || (defined( FLM_WIN) && !defined( FLM_64BIT))

	FastBlockCheckSum( pucHdr, &uiAdds, &uiXORs,
		(unsigned long)sizeof( SFLM_DB_HDR));

#else
	FLMBYTE *		pucCur = pucHdr;
	FLMBYTE *		pucEnd = pucHdr + sizeof( SFLM_DB_HDR);

	while( pucCur < pucEnd)	
	{
		uiAdds += *pucCur;
		uiXORs ^= *pucCur++;
	}

	uiAdds &= 0xFF;
#endif

	// Restore the checksum that was in the header

	pDbHdr->ui32HdrCRC = ui32SaveCRC;
	return( (FLMUINT32)((uiAdds << 16) + uiXORs));
}
