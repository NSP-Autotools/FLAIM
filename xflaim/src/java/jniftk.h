//------------------------------------------------------------------------------
// Desc:
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

#ifndef JNIFTK_H
#define JNIFTK_H

	#ifndef FLM_DONT_USE_COM
		#define FLM_DONT_USE_COM
	#endif
	
	#include <xflaim.h>
	#include <flaimtk.h>
	#include "jni.h"
	
	void ThrowError(
		RCODE			rc,
		JNIEnv *		pEnv);
	
	RCODE getUniString(
		JNIEnv *		pEnv,
		jstring		sStr,
		F_DynaBuf *	pDynaBuf);
		
	RCODE getUTF8String(
		JNIEnv *		pEnv,
		jstring		sStr,
		F_DynaBuf *	pDynaBuf);
	
#endif // JNIFTK_H
