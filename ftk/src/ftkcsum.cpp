//------------------------------------------------------------------------------
// Desc:	This file contains routines which calculates checksums
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

#include "ftksys.h"

static FLMUINT32 *		gv_pui32CRCTbl = NULL;
#if defined( FLM_X86) && (defined( FLM_GNUC) || defined( FLM_WIN) || defined( FLM_NLM))
static FLMBOOL				gv_bCanUseFastCheckSum = FALSE;
#endif

extern "C" void ftkFastChecksum(
	const void *			pBlk,
	unsigned long *		puiSum,	
	unsigned long *		puiXOR,
	unsigned long			uiNumberOfBytes);
		
extern "C" unsigned long ftkGetMMXSupported(void);

#if defined( FLM_X86)
	#if defined( FLM_GNUC) || (defined( FLM_WIN) && !defined(FLM_64BIT)) || defined( FLM_NLM)
		#define FLM_HAVE_FAST_CHECKSUM_ROUTINES
	#endif
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_WATCOM_NLM) && defined( FLM_RING_ZERO_NLM)
	#pragma aux ftkGetMMXSupported parm;
	#pragma aux ftkGetMMXSupported = \
		0xB8 0x01 0x00 0x00 0x00            /* mov		eax, 1  				*/\
		0x0F 0xA2                         	/* CPUID  							*/\
		0x33 0xC0                         	/* xor		eax, eax 			*/\
		0xF7 0xC2 0x00 0x00 0x80 0x00       /* test		edx, (1 SHL 23) 	*/\
		0x0F 0x95 0xC0                      /* setnz	al  						*/\
		modify exact [EAX EBX ECX EDX];
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_WATCOM_NLM) && defined( FLM_LIBC_NLM)
unsigned long ftkGetMMXSupported( void)
{
	return( 1);
}
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_WIN) && !defined( FLM_64BIT)
unsigned long ftkGetMMXSupported( void)
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
	
	return( bMMXSupported);
}
#endif
	
/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_32BIT) && defined( FLM_GNUC)
unsigned long ftkGetMMXSupported( void)
{
	FLMUINT32 	bMMXSupported;
	
	__asm__ __volatile__(
		"push		%%ebx\n"
		"mov		$1, %%eax\n"
		"cpuid\n"
		"xor		%%eax, %%eax\n"
		"test		$0x800000, %%edx\n"
		"setnz	%%al\n"
		"mov		%%eax, %0\n"
		"pop		%%ebx\n"
			: "=&r" (bMMXSupported)
			:
			: "%eax", "%ecx", "%edx");
	
	return( bMMXSupported);
}
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_64BIT)
unsigned long ftkGetMMXSupported( void)
{
	return( 1);
}
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_WATCOM_NLM)

	#pragma aux ftkFastChecksum parm [ESI] [eax] [ebx] [ecx];
	#pragma aux ftkFastChecksum = \
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
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_32BIT) && defined( FLM_WIN)
void ftkFastChecksum(
	const void *		pBlk,
	unsigned long *	puiSum,	
	unsigned long *	puiXOR,
	unsigned long		uiNumberOfBytes)
{
	__asm
	{
			mov		esi, pBlk

			// Load up the starting checksum values into edx (add) and ebx (XOR)

			mov		eax, puiSum
			mov		edx, [eax]
			and		edx, 0ffh			;clear unneeded bits 
			mov		eax, puiXOR
			mov		ebx, [eax]
			and		ebx, 0ffh			;clear unneeded bits 
			mov		ecx, uiNumberOfBytes
			mov		edi, ecx				;save the amount to copy 

			cmp		ecx, 32				;see if we have enough for the big loop 
			jb			MediumStuff 					

			shr		ecx, 5				;convert length to 32 byte blocks
			and		edi, 01fh			;change saved length to remainder
			
			movd		mm4, edx				;set ADD
			movd		mm5, ebx				;set XOR

BigStuffLoop:
												;load up mm0 - mm3 with 8 bytes each of data.
			movq		mm0, [esi]
			movq		mm1, [esi + 8]
			movq		mm2, [esi + 16]
			movq		mm3, [esi + 24]
			add		esi, 32				;move the data pointer ahead 32

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

			movd		ebx, mm5
			psrlq		mm5, 32 
			movd		eax, mm5
			xor		ebx, eax
												 
			movq		mm0, mm4				;extract the sum value from mm4 and put it in dl & dh
			psrlq		mm0, 32 
			paddb		mm4, mm0
			movq		mm0, mm4
			psrlq		mm0, 16 
			paddb		mm4, mm0
			movd		edx, mm4
			
			emms								;end of MMX stuff

			mov		ecx, edi				;load up the rest of the length
			
MediumStuff:
			cmp		ecx, 4
			jb			SmallStuff
			shr		ecx, 2
			and		edi, 3

MediumStuffLoop:
			mov		eax, [esi]
			add		esi, 4
			xor		ebx, eax
			add		dl, al
			add		dh, ah
			shr		eax, 16
			add		dl, al
			add		dh, ah
			dec		ecx
			jnz		MediumStuffLoop;
			mov		ecx, edi				;load up the rest of the length
			
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

			mov		eax, puiSum
			mov		[eax], edx

			mov		eax, puiXOR
			mov		[eax], ebx
	}
	return;
}
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_32BIT) && defined( FLM_GNUC)
void ftkFastChecksum(
		const void *		pBlk,
		unsigned long *	puiSum,	
		unsigned long *	puiXOR,
		unsigned long		uiNumberOfBytes)
{
	__asm__ __volatile__(
			"			push		%%ebx\n"
			"			mov		%2, %%esi\n"
			"			mov		%3, %%eax\n"
			"			mov		(%%eax), %%edx\n"
			"			and		$0xFF, %%edx\n"
			"			mov		%4, %%eax\n"
			"			mov		(%%eax), %%ebx\n"
			"			and		$0xFF, %%ebx\n" 
			"			mov		%5, %%ecx\n"
			"			mov		%%ecx, %%edi\n" 
			
			"			cmp		$32, %%ecx\n"
			"			jb			2f # MediumStuff\n" 					
			
			"			shr		$5, %%ecx\n"
			"			and		$0x01F, %%edi\n"
						
			"			movd		%%edx, %%mm4\n"
			"			movd		%%ebx, %%mm5\n"
			
			"1: # BigStuffLoop:\n"
			"			movq		(%%esi), %%mm0\n"
			"			movq		8(%%esi), %%mm1\n"
			"			movq		16(%%esi), %%mm2\n"
			"			movq		24(%%esi), %%mm3\n"
			"			add		$32, %%esi\n"
			"			paddb		%%mm0, %%mm4\n"
			"			pxor		%%mm0, %%mm5\n"
			"			paddb		%%mm1, %%mm4\n"
			"			pxor		%%mm1, %%mm5\n"
			"			paddb		%%mm2, %%mm4\n"
			"			pxor		%%mm2, %%mm5\n"
			"			paddb		%%mm3, %%mm4\n"
			"			pxor		%%mm3, %%mm5\n"
			"			dec		%%ecx\n"
			
			"			jnz		1b # BigStuffLoop\n" 
			"			movd		%%mm5, %%ebx\n"
			"			psrlq		$32, %%mm5\n"
			"			movd		%%mm5, %%eax\n"
			"			xor		%%eax, %%ebx\n"
			"			movq		%%mm4, %%mm0\n"
			"			psrlq		$32, %%mm0\n"
			"			paddb		%%mm0, %%mm4\n"
			"			movq		%%mm4, %%mm0\n"
			"			psrlq		$16, %%mm0\n" 
			"			paddb		%%mm0, %%mm4\n"
			"			movd		%%mm4, %%edx\n"
			"			emms\n"
			
			"			mov		%%edi, %%ecx\n"
			
			"2: # MediumStuff:\n"
			"			cmp		$4, %%ecx\n"
			"			jb			4f # SmallStuff\n"
			"			shr		$2, %%ecx\n"
			"			and		$3, %%edi\n"
			
			"3: # MediumStuffLoop:\n"
			"			mov		(%%esi), %%eax\n"
			"			add		$4, %%esi\n"
			"			xor		%%eax, %%ebx\n"
			"			add		%%al, %%dl\n"
			"			add		%%ah, %%dh\n"
			"			shr		$16, %%eax\n"
			"			add		%%al, %%dl\n"
			"			add		%%ah, %%dh\n"
			"			dec		%%ecx\n"
			"			jnz		3b # MediumStuffLoop\n"
			"			mov		%%edi, %%ecx\n"
			
			"4: # SmallStuff:\n"
			"			add		%%dh, %%dl\n" 
			"			mov		%%ebx, %%eax\n"
			"			shr		$16, %%eax\n"				
			"			xor		%%ax, %%bx\n" 							
			"			xor		%%bh, %%bl\n"							
			"			cmp		$0, %%ecx\n" 
			"			jz			6f # Done\n" 							
			
			"5: # SmallStuffLoop:\n" 						
			"			mov		(%%esi), %%al\n" 						
			"			inc		%%esi\n"								
			"			add		%%al, %%dl\n" 							
			"			xor		%%al, %%bl\n" 							
			"			dec		%%ecx\n" 							
			"			jnz		5b # SmallStuffLoop\n" 				
			"6: # Done:\n" 									
			"			and		$0xFF, %%edx\n" 
			"			and		$0xFF, %%ebx\n" 
			
			"			mov		%0, %%eax\n"
			"			mov		%%edx, (%%eax)\n"
			
			"			mov		%1, %%eax\n"
			"			mov		%%ebx, (%%eax)\n"
			"			pop		%%ebx\n"
				: "=m" (puiSum), "=m" (puiXOR)
				: "m" (pBlk), "m" (puiSum), "m" (puiXOR), "m" (uiNumberOfBytes)
				: "%eax", "%ecx", "%edx", "%esi", "%edi");
}
#endif

/********************************************************************
Desc:
*********************************************************************/
#if defined( FLM_X86) && defined( FLM_64BIT) && defined( FLM_GNUC)
void ftkFastChecksum(
		const void *		pBlk,
		unsigned long *	puiSum,	
		unsigned long *	puiXOR,
		unsigned long		uiNumberOfBytes)
{
	__asm__ __volatile__(
			"			mov		%2, %%r8\n"
			"			mov		%3, %%r9\n"
			"			mov		(%%r9), %%edx\n"
			"			and		$0xFF, %%edx\n"
			"			mov		%4, %%r9\n"
			"			mov		(%%r9), %%ebx\n"
			"			and		$0xFF, %%ebx\n" 
			"			mov		%5, %%ecx\n"
			"			mov		%%ecx, %%edi\n" 
			
			"			cmp		$32, %%ecx\n"
			"			jb			2f # MediumStuff\n" 					
			
			"			shr		$5, %%ecx\n"
			"			and		$0x01F, %%edi\n"
						
			"			movd		%%edx, %%mm4\n"
			"			movd		%%ebx, %%mm5\n"
			
			"1: # BigStuffLoop:\n"
			"			movq		(%%r8), %%mm0\n"
			"			movq		8(%%r8), %%mm1\n"
			"			movq		16(%%r8), %%mm2\n"
			"			movq		24(%%r8), %%mm3\n"
			"			add		$32, %%r8\n"
			"			paddb		%%mm0, %%mm4\n"
			"			pxor		%%mm0, %%mm5\n"
			"			paddb		%%mm1, %%mm4\n"
			"			pxor		%%mm1, %%mm5\n"
			"			paddb		%%mm2, %%mm4\n"
			"			pxor		%%mm2, %%mm5\n"
			"			paddb		%%mm3, %%mm4\n"
			"			pxor		%%mm3, %%mm5\n"
			"			dec		%%ecx\n"
			
			"			jnz		1b # BigStuffLoop\n" 
			"			movd		%%mm5, %%ebx\n"
			"			psrlq		$32, %%mm5\n"
			"			movd		%%mm5, %%eax\n"
			"			xor		%%eax, %%ebx\n"
			"			movq		%%mm4, %%mm0\n"
			"			psrlq		$32, %%mm0\n"
			"			paddb		%%mm0, %%mm4\n"
			"			movq		%%mm4, %%mm0\n"
			"			psrlq		$16, %%mm0\n" 
			"			paddb		%%mm0, %%mm4\n"
			"			movd		%%mm4, %%edx\n"
			"			emms\n"
			
			"			mov		%%edi, %%ecx\n"
			
			"2: # MediumStuff:\n"
			"			cmp		$4, %%ecx\n"
			"			jb			4f # SmallStuff\n"
			"			shr		$2, %%ecx\n"
			"			and		$3, %%edi\n"
			
			"3: # MediumStuffLoop:\n"
			"			mov		(%%r8), %%eax\n"
			"			add		$4, %%r8\n"
			"			xor		%%eax, %%ebx\n"
			"			add		%%al, %%dl\n"
			"			add		%%ah, %%dh\n"
			"			shr		$16, %%eax\n"
			"			add		%%al, %%dl\n"
			"			add		%%ah, %%dh\n"
			"			dec		%%ecx\n"
			"			jnz		3b # MediumStuffLoop\n"
			"			mov		%%edi, %%ecx\n"
			
			"4: # SmallStuff:\n"
			"			add		%%dh, %%dl\n" 
			"			mov		%%ebx, %%eax\n"
			"			shr		$16, %%eax\n"				
			"			xor		%%ax, %%bx\n" 							
			"			xor		%%bh, %%bl\n"							
			"			cmp		$0, %%ecx\n" 
			"			jz			6f # Done\n" 							
			
			"5: # SmallStuffLoop:\n" 						
			"			mov		(%%r8), %%al\n" 						
			"			inc		%%r8\n"								
			"			add		%%al, %%dl\n"
			"			xor		%%al, %%bl\n" 							
			"			dec		%%ecx\n" 							
			"			jnz		5b # SmallStuffLoop\n" 				
			"6: # Done:\n" 									
			"			and		$0xFF, %%edx\n" 
			"			and		$0xFF, %%ebx\n" 
			
			"			mov		%0, %%r9\n"
			"			mov		%%edx, (%%r9)\n"
			
			"			mov		%1, %%r9\n"
			"			mov		%%ebx, (%%r9)\n"
				: "=m" (puiSum), "=m" (puiXOR)
				: "m" (pBlk), "m" (puiSum), "m" (puiXOR), "m" (uiNumberOfBytes)
				: "%eax", "%ebx", "%ecx", "%edi", "%edx", "%r8", "%r9");
}
#endif

/****************************************************************************
Desc:
****************************************************************************/
#ifndef FLM_HAVE_FAST_CHECKSUM_ROUTINES
void ftkFastChecksum(
	const void *,		// pBlk,
	unsigned long *,	// puiSum,	
	unsigned long *,	// puiXOR,
	unsigned long)		// uiNumberOfBytes)
{
	f_assert( 0);
}
#endif

/******************************************************************************
Desc: Sets the global variable to check if MMX instructions are allowed.
******************************************************************************/
void f_initFastCheckSum( void)
{
#if defined( FLM_X86) && (defined( FLM_GNUC) || defined( FLM_WIN) || defined( FLM_NLM))
	// NOTE that ftkGetMMXSupported assumes that we are running on at least a
	// pentium.  The check to see if we are on a pentium requires that  we
	// modify the flags register, and we can't do that if we are running
	// in ring3.  Because NetWare 5 - according to our product marketing -
	// requires at least a P5 90Mhz, we will be safe.  When you port this
	// code to NT, you may need to come up with a safe way to see if we
	// can do MMX instructions - unless you can assume that even on NT you
	// will be on at least a P5.

	gv_bCanUseFastCheckSum = ftkGetMMXSupported() ? TRUE : FALSE;
#endif
}

/********************************************************************
Desc:	Calculate the checksum for a block.  NOTE: This is ALWAYS done
		on the raw image that will be written to disk.  This means
		that if the block needs to be converted before writing it out,
		it should be done before calculating the checksum.
*********************************************************************/
FLMUINT32 FTKAPI f_calcFastChecksum(
	const void *	pvData,
	FLMUINT			uiLength,
	FLMUINT *		puiSum,
	FLMUINT *		puiXOR)
{
	FLMUINT			uiSum = 0;
	FLMUINT			uiXOR = 0;
	FLMBYTE *		pucData = (FLMBYTE *)pvData;
	
	if( puiSum)
	{
		uiSum = *puiSum;
	}
	
	if( puiXOR)
	{
		uiXOR = *puiXOR;
	}

#ifdef FLM_HAVE_FAST_CHECKSUM_ROUTINES
	if( gv_bCanUseFastCheckSum)
	{
		ftkFastChecksum( pvData, (unsigned long *) &uiSum, 
					(unsigned long *) &uiXOR, (unsigned long) uiLength);
	}
	else
#endif
	{
		register FLMBYTE *	pucCur = pucData;
		register FLMBYTE *	pucEnd = pucData + uiLength;
	
		while( pucCur < pucEnd)	
		{
		 	uiSum += *pucCur;
			uiXOR ^= *pucCur++;
		}

		uiSum &= 0xFF;
	}

	if( puiSum)
	{
		*puiSum = uiSum;
	}
	
	if( puiXOR)
	{
		*puiXOR = uiXOR;
	}
	
	return( (FLMUINT32)((uiSum << 16) + uiXOR));
}

/****************************************************************************
Desc: Generates a table of remainders for each 8-bit byte.  The resulting
		table is used by f_updateCRC to calculate a CRC value.  The table
		must be freed via a call to f_free.
*****************************************************************************/
RCODE f_initCRCTable( void)
{
	RCODE				rc = NE_FLM_OK;
	FLMUINT32 *		pTable;
	FLMUINT32		ui32Val;
	FLMUINT32		ui32Loop;
	FLMUINT32		ui32SubLoop;

	// Use the standard degree-32 polynomial used by
	// Ethernet, PKZIP, etc. for computing the CRC of
	// a data stream.  This is the little-endian
	// representation of the polynomial.  The big-endian
	// representation is 0x04C11DB7.

#define CRC_POLYNOMIAL		((FLMUINT32)0xEDB88320)

	f_assert( !gv_pui32CRCTbl);

	if( RC_BAD( rc = f_alloc( 256 * sizeof( FLMUINT32), &pTable)))
	{
		goto Exit;
	}

	for( ui32Loop = 0; ui32Loop < 256; ui32Loop++)
	{
		ui32Val = ui32Loop;
		for( ui32SubLoop = 0; ui32SubLoop < 8; ui32SubLoop++)
		{
			if( ui32Val & 0x00000001)
			{
				ui32Val = CRC_POLYNOMIAL ^ (ui32Val >> 1);
			}
			else
			{
				ui32Val >>= 1;
			}
		}

		pTable[ ui32Loop] = ui32Val;
	}

	gv_pui32CRCTbl = pTable;
	pTable = NULL;

Exit:

	if( pTable)
	{
		f_free( &pTable);
	}

	return( rc);
}

/****************************************************************************
Desc:
*****************************************************************************/
void f_freeCRCTable( void)
{
	if( gv_pui32CRCTbl)
	{
		f_free( &gv_pui32CRCTbl);
	}
}
	
/****************************************************************************
Desc: Computes the CRC of the passed-in data buffer.  Multiple calls can
		be made to this routine to build a CRC over multiple data buffers.
		On the first call, *pui32CRC must be initialized to something
		(0, etc.).  For generating CRCs that are compatible with PKZIP,
		*pui32CRC should be initialized to 0xFFFFFFFF and the ones complement
		of the resulting CRC should be computed.
*****************************************************************************/
void FTKAPI f_updateCRC(
	const void *		pvBuffer,
	FLMUINT				uiCount,
	FLMUINT32 *			pui32CRC)
{
	FLMBYTE *			pucBuffer = (FLMBYTE *)pvBuffer;
	FLMUINT32			ui32CRC = *pui32CRC;
	FLMUINT				uiLoop;

	for( uiLoop = 0; uiLoop < uiCount; uiLoop++)
	{
		ui32CRC = (ui32CRC >> 8) ^ gv_pui32CRCTbl[
			((FLMBYTE)(ui32CRC & 0x000000FF)) ^ pucBuffer[ uiLoop]];
	}

	*pui32CRC = ui32CRC;
}

/********************************************************************
Desc:
*********************************************************************/
FLMBYTE FTKAPI f_calcPacketChecksum(
	const void *	pvPacket,
	FLMUINT			uiBytesToChecksum)
{
	FLMUINT			uiChecksum = 0;
	
#ifdef FLM_HAVE_FAST_CHECKSUM_ROUTINES
	if( gv_bCanUseFastCheckSum)
	{
		FLMUINT		uiSum;
		
		ftkFastChecksum( pvPacket, (unsigned long *) &uiSum, 
				(unsigned long *) &uiChecksum, (unsigned long) uiBytesToChecksum);
	}
	else
#endif
	{	
		FLMBYTE *		pucEnd;
		FLMBYTE *		pucSectionEnd;
		FLMBYTE *		pucCur;
		FLMBYTE			ucTmp;
			
		pucCur = (FLMBYTE *)pvPacket;
		pucEnd = pucCur + uiBytesToChecksum;
	
	#ifdef FLM_64BIT
		pucSectionEnd = pucCur + (sizeof( FLMUINT) - ((FLMUINT)pucCur & 0x7));
	#else
		pucSectionEnd = pucCur + (sizeof( FLMUINT) - ((FLMUINT)pucCur & 0x3));
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
		uiChecksum = (FLMUINT)ucTmp;
	}

	return( (FLMBYTE)(uiChecksum != 0 ? uiChecksum : 1));
}
