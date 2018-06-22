//------------------------------------------------------------------------------
// Desc:	Stream tests
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
	// Stream tests
	//--------------------------------------------------------------------------
	public class StreamTests : Tester
	{
		private const string TEST_STREAM_STRING = "abcdefghijklmnopqrstuvwxyzABCDEFJHIJKLMNOPQRSTUVWXYZ0123456789";

		public bool streamTests(
			DbSystem	dbSystem)
		{
			IStream			bufferStream;
			IStream			encoderStream;
			IStream			decoderStream;
			OStream			fileOStream;
			Stream			s;
			StreamReader	sr;
			string			sFileData;

			beginTest( "Creating IStream from buffer");
			try
			{
				bufferStream = dbSystem.openBufferIStream( TEST_STREAM_STRING);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling openBufferIStream");
				return( false);
			}
			endTest( false, true);

			beginTest( "Creating base 64 encoder stream");
			try
			{
				encoderStream = dbSystem.openBase64Encoder( bufferStream, true);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling openBase64Encoder");
				return( false);
			}
			endTest( false, true);

			beginTest( "Creating base 64 decoder stream");
			try
			{
				decoderStream = dbSystem.openBase64Decoder( encoderStream);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling openBase64Decoder");
				return( false);
			}
			endTest( false, true);

			beginTest( "Creating file output stream");
			try
			{
				fileOStream = dbSystem.openFileOStream( "Output_Stream", true);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling openFileOStream");
				return( false);
			}
			endTest( false, true);

			beginTest( "Writing from input stream to output stream");
			try
			{
				dbSystem.writeToOStream( decoderStream, fileOStream);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling writeToOStream");
				return( false);
			}
			fileOStream.close();
			endTest( false, true);

			beginTest( "Comparing output stream data to original data");

			s = File.OpenRead( "Output_Stream");
			sr = new StreamReader( s);
			sFileData = sr.ReadLine();
			if (sFileData != TEST_STREAM_STRING)
			{
				endTest( false, false);
				System.Console.WriteLine( "Stream data does not match original string");
				System.Console.WriteLine( "File Data:\n[{0}]", sFileData);
				System.Console.WriteLine( "Original String:\n[{0}]", TEST_STREAM_STRING);
				return( false);
			}

			endTest( false, true);
			return( true);
		}
	}
}
