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

import java.util.Collections;
import xflaim.*;

public class XFlaimTester
{

	public static void main(String[] args) throws XFlaimException
	{
		DbSystem					dbSystem = null;
		CREATEOPTS				createOpts = null;
		Db							jDb = null;
		IStream					jIStream = null;
		boolean					bDone = false;
		int						iCount = 0;
		int						iBufferSize;
		boolean					bCreate = false;
		DOMNode					jDoc = null;

		try
		{
			// Initialize the dbSystem so we can do a few things.
			
			if (dbSystem == null)
			{
				dbSystem = new DbSystem();
			}
		}
		catch (XFlaimException e)
		{
			System.out.println( "Couldn't create the DbSystem.  Exception message: " +
				e.getMessage());
			System.exit( 0);
		}
			
		while (!bDone)
		{	
			try
			{
				jDb = dbSystem.dbOpen("tst.db", null, null, null, true);
				bDone = true;
				System.out.println("Database successfully opened!");
			}
			catch (XFlaimException e)
			{
				bCreate = true;
			}

			try
			{
				// Try to create it.
				
				if (bCreate)
				{
				
					jDb = dbSystem.dbCreate( "tst.db", "", "", "", "", null);
					bDone = true;
					System.out.println("Database successfully created!");
				}
			}
			catch (XFlaimException e)
			{
				System.out.println( "Database could not be created.  Attempting to remove.");
				if (iCount < 5)
				{
					dbSystem.dbRemove("tst.db", "", "", true);
					iCount++;				
				}
				else
				{
					e.printStackTrace();
					System.exit( 0);	
				}
			}
		}
		
		jDb.close();
		jDb = dbSystem.dbOpen("tst.db", null, null, null, true);
			
		System.out.println("Database successfully re-opened!");
		
		// Open a BufferIstream (just for fun)
		
		String str = "<?xml version=\"1.0\"?>" +
						  "<disc>" +
						  "	<id>7004df09</id>" +
						  "	<length>1249</length>" +
						  "	<title>Last of the Juanitas / Time's Up</title>" +
						  "	<genre>cddb/rock</genre>" +
						  "	<track index=\"1\" offset=\"150\">Of Course - Nowadays - They Call It Stalking</track>" +
						  "	<track index=\"2\" offset=\"13661\">Make You Cry</track>" +
						  "	<track index=\"3\" offset=\"26829\">Look Bolt the Door</track>" +
						  "	<track index=\"4\" offset=\"34269\">Here It Comes</track>" +
						  "	<track index=\"5\" offset=\"46567\">Time's Up</track>" +
						  "	<track index=\"6\" offset=\"52488\">Big Eyed Space Girl</track>" +
						  "</disc>";

		try
		{
			jIStream = dbSystem.openBufferIStream( str);
			System.out.println("Created a BufferedIStream");
		}
		catch (XFlaimException e)
		{
			// Can't go any farther.
			System.out.println( "Caught openBufferIStream exception: " + e.getMessage());
			e.printStackTrace();
		}

		// Now import a document.
		
		try
		{
			// Begin a transaction
			
			jDb.transBegin( TransactionType.UPDATE_TRANS, 0, 0);
										
			System.out.println("Began an UPDATE transaction");

			jDb.Import(jIStream, xflaim.Collections.DATA);
			
			System.out.println("Imported the document");
			
			jDb.transCommit();
			
			System.out.println("Committed the transaction");
		}
		catch (XFlaimException e)
		{
			System.out.println( "Caught XFlaim exception: " + e.getMessage());			
			e.printStackTrace();
		}
		
		// Get the first document.
		
		try
		{
			jDb.transBegin( TransactionType.READ_TRANS, 0, 0);
			jDoc = jDb.getFirstDocument( xflaim.Collections.DATA, null);
			jDb.transCommit();
			jDoc.release();
			System.out.println("Got the first document.");
		}
		catch (XFlaimException e)
		{
			System.out.println("Caught getFirstDocument exception: " + e.getMessage());
			e.printStackTrace();
		}
		
		// Now shut down.
		
		System.out.println("Shutting down");
		jIStream.release();
		jDb.close();
		dbSystem.dbClose();
	}
}
