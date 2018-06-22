//------------------------------------------------------------------------------
// Desc:	DataVector tests
// Tabs:	3
//
// Copyright (c) 2006-2007 Novell, Inc. All Rights Reserved.
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
using System.IO;
using System.Runtime.InteropServices;
using xflaim;

namespace cstest
{

	//--------------------------------------------------------------------------
	// Vector tests
	//--------------------------------------------------------------------------
	public class VectorTests : Tester
	{
		public bool vectorTests(
			DbSystem	dbSystem)
		{
			DataVector	v;
			string		setString = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
			string		getString = "XXX";
			byte []		setBinary = new byte [] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
			byte []		getBinary = new byte [] {0};
			bool			bDataSame;
			ulong			setULong = 255873421849;
			ulong			getULong = 0;
			long			setLong = -234887;
			long			getLong = 0;
			int			setInt = -400;
			int			getInt = 0;
			uint			setUInt = 880044;
			uint			getUInt = 0;

			beginTest( "Creating DataVector");
			try
			{
				v = dbSystem.createDataVector();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling createDataVector");
				return( false);
			}
			endTest( false, true);

			// Test setting and getting of binary data

			beginTest( "Setting binary data");
			try
			{
				v.setBinary( 0, setBinary);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setBinary");
				return( false);
			}
			endTest( false, true);

			beginTest( "Getting binary data");
			try
			{
				getBinary = v.getBinary( 0);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getBinary");
				return( false);
			}
			endTest( false, true);


			beginTest( "Comparing set binary data to get binary data");

			bDataSame = true;
			if (setBinary.Length != getBinary.Length)
			{
				bDataSame = false;
			}
			else
			{
				for( uint uiLoop = 0; uiLoop < setBinary.Length; uiLoop++)
				{
					if (setBinary [uiLoop] != getBinary [uiLoop])
					{
						bDataSame = false;
						break;
					}
				}
			}
			if (!bDataSame)
			{
				endTest( false, false);
				System.Console.WriteLine( "Set binary data does not match get binary data");
				System.Console.Write( "Set Binary Data Length: {0}\n[", setBinary.Length);
				for( uint uiLoop = 0; uiLoop < setBinary.Length; uiLoop++)
				{
					System.Console.Write( "{0} ", setBinary[uiLoop]);
				}
				System.Console.WriteLine( "]");
				System.Console.Write( "Get Binary Data Length: {0}\n[", getBinary.Length);
				for( uint uiLoop = 0; uiLoop < getBinary.Length; uiLoop++)
				{
					System.Console.Write( "{0} ", getBinary[uiLoop]);
				}
				System.Console.WriteLine( "]");
				return( false);
			}
			endTest( false, true);

			// Test setting and getting of string data

			beginTest( "Setting string data");
			try
			{
				v.setString( 1, setString);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setString");
				return( false);
			}
			endTest( false, true);

			beginTest( "Getting string data");
			try
			{
				getString = v.getString( 1);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getString");
				return( false);
			}
			endTest( false, true);


			beginTest( "Comparing set string data to get string data");

			if (setString != getString)
			{
				endTest( false, false);
				System.Console.WriteLine( "Set string data does not match get string data");
				System.Console.WriteLine( "Set String:\n[{0}]", setString);
				System.Console.WriteLine( "Get String:\n[{0}]", getString);
			}
			endTest( false, true);

			// Test setting and getting of ulong data

			beginTest( "Setting ulong data");
			try
			{
				v.setULong( 2, setULong);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setULong");
				return( false);
			}
			endTest( false, true);

			beginTest( "Getting ulong data");
			try
			{
				getULong = v.getULong( 2);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getULong");
				return( false);
			}
			endTest( false, true);


			beginTest( "Comparing set ulong data to get ulong data");

			if (setULong != getULong)
			{
				endTest( false, false);
				System.Console.WriteLine( "Set ulong data does not match get ulong data");
				System.Console.WriteLine( "Set: {0}, Get: {1}", setULong, getULong);
			}
			endTest( false, true);

			// Test setting and getting of long data

			beginTest( "Setting long data");
			try
			{
				v.setLong( 3, setLong);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setLong");
				return( false);
			}
			endTest( false, true);

			beginTest( "Getting long data");
			try
			{
				getLong = v.getLong( 3);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getLong");
				return( false);
			}
			endTest( false, true);


			beginTest( "Comparing set long data to get long data");

			if (setLong != getLong)
			{
				endTest( false, false);
				System.Console.WriteLine( "Set long data does not match get long data");
				System.Console.WriteLine( "Set: {0}, Get: {1}", setLong, getLong);
			}
			endTest( false, true);

			// Test setting and getting of uint data

			beginTest( "Setting uint data");
			try
			{
				v.setUInt( 4, setUInt);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setUInt");
				return( false);
			}
			endTest( false, true);

			beginTest( "Getting uint data");
			try
			{
				getUInt = v.getUInt( 4);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getUInt");
				return( false);
			}
			endTest( false, true);


			beginTest( "Comparing set uint data to get uint data");

			if (setUInt != getUInt)
			{
				endTest( false, false);
				System.Console.WriteLine( "Set uint data does not match get uint data");
				System.Console.WriteLine( "Set: {0}, Get: {1}", setUInt, getUInt);
			}
			endTest( false, true);

			// Test setting and getting of int data

			beginTest( "Setting int data");
			try
			{
				v.setInt( 5, setInt);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling setInt");
				return( false);
			}
			endTest( false, true);

			beginTest( "Getting int data");
			try
			{
				getInt = v.getInt( 5);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling getInt");
				return( false);
			}
			endTest( false, true);


			beginTest( "Comparing set int data to get int data");

			if (setInt != getInt)
			{
				endTest( false, false);
				System.Console.WriteLine( "Set int data does not match get int data");
				System.Console.WriteLine( "Set: {0}, Get: {1}", setInt, getInt);
			}
			endTest( false, true);

			return( true);
		}
	}
}
