//------------------------------------------------------------------------------
// Desc:	Import tests
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
	// Open database test.
	//--------------------------------------------------------------------------
	public class ImportTests : Tester
	{
		private const string sDoc1 = 
			"<?xml version=\"1.0\"?>" +
			"<disc> " +
		"<id>00054613</id> " +
		"<length>1352</length> " +
		"<title>Margret Birkenfeld / Zachaus</title> " +
		"<genre>cddb/misc</genre> " +
		"<ext> ID3G: 77</ext> " +
		"<track index=\"1\" offset=\"150\">Wie erzahlen euch 1. Srophe </track> " +
		"<track index=\"2\" offset=\"13065\">Wir erzahlen Euch 2. Strophe </track> " +
		"<track index=\"3\" offset=\"14965\">Zachaus ist ein reicher Mann 1+2 Str</track> " +
		"<track index=\"4\" offset=\"19980\">Jericho</track> " +
		"<track index=\"5\" offset=\"28122\">Haruck, schnauf schnauf 1+2 Strophe</track> " +
		"<track index=\"6\" offset=\"33630\">Haruck, schnauf schnauf 3 Strophe</track> " +
		"<track index=\"7\" offset=\"37712\">Zachaus ist ein reicher Mann 3. Stophe</track> " +
		"<track index=\"8\" offset=\"41502\">Zachaus komm herunter!</track> " +
		"<track index=\"9\" offset=\"57627\">Wir erzahlen euch</track> " +
		"<track index=\"10\" offset=\"63145\">Leer ab jetzt Playback</track> " +
		"<track index=\"11\" offset=\"65687\">Wie erzahlen euch 1. Srophe Pb</track> " +
		"<track index=\"12\" offset=\"69212\">Wir erzahlen Euch 2. Strophe Pb</track> " +
		"<track index=\"13\" offset=\"71102\">Zachaus ist ein reicher Mann 1+2 Str Pb</track> " +
		"<track index=\"14\" offset=\"75622\">Jericho Pb</track> " +
		"<track index=\"15\" offset=\"82292\">Haruck, schnauf schnauf 1+2 Strophe Pb</track> " +
		"<track index=\"16\" offset=\"86555\">Haruck, schnauf schnauf 3 Strophe Pb</track> " +
		"<track index=\"17\" offset=\"89887\">Zachaus ist ein reicher Mann 3. Stophe Pb</track> " +
		"<track index=\"18\" offset=\"93067\">Zachaus komm herunter! Pb</track> " +
		"<track index=\"19\" offset=\"97797\">Wir erzahlen euch Pb</track> " +
		"</disc> ";

		private const string sDoc2 = "<?xml version=\"1.0\"?> " +
			"<disc> " +
		"<id>0008a40f</id> " +
		"<length>2214</length> " +
		"<title>rundu... - Visur Ur Vinsabokinni</title> " +
		"<genre>cddb/misc</genre> " +
		"<track index=\"1\" offset=\"150\">Blessuo Solin Elskar Allt - Ur Augum Stirur Strjukio Fljott</track> " +
		"<track index=\"2\" offset=\"13855\">Heioloarkvaeoi</track> " +
		"<track index=\"3\" offset=\"27576\">Buxur, Vesti, Brok og Sko</track> " +
		"<track index=\"4\" offset=\"33311\">Gekk Eg Upp A Holinn</track>" +
		"<track index=\"5\" offset=\"45340\">Nu Blanar Yfir Berjamo - A Berjamo</track> " +
		"<track index=\"6\" offset=\"59209\">Orninn Flygur Fugla Haest - Solskrikjan - Min</track> " +
		"<track index=\"7\" offset=\"64309\">Nu Er Glatt I Borg Og Bae</track> " +
		"<track index=\"8\" offset=\"73018\">Smaladrengurinn - Klappa Saman Lofunum</track> " +
		"<track index=\"9\" offset=\"89149\">Stigur Hun Vio Stokkinn</track> " +
		"<track index=\"10\" offset=\"91370\">Dansi, Dansi, Dukkan Min</track> " +
		"<track index=\"11\" offset=\"104540\">Rioum Heim Til Hola - Gott Er Ao Rioa Sandana Mjuka</track> " +
		"<track index=\"12\" offset=\"119232\">Gryla - Jolasveinar Ganga Um Golf</track> " +
		"<track index=\"13\" offset=\"133837\">Erla, Gooa Erla</track> " +
		"<track index=\"14\" offset=\"146208\">Vio Skulum Ekki Hafa Hatt</track> " +
		"<track index=\"15\" offset=\"149899\">Sofa Urtu Born</track> " +
		"</disc> ";

		private const string sIndexDef = "<xflaim:Index " +
			"	xmlns:xflaim=\"http://www.novell.com/XMLDatabase/Schema\"" +
		"	xflaim:name=\"Title+Index+Offset\" " +
		"	xflaim:DictNumber=\"1\"> " +
		"	<xflaim:ElementComponent " +
		"		xflaim:name=\"title\" " +
		"		xflaim:KeyComponent=\"1\" " +
		"		xflaim:IndexOn=\"value\"/> " +
		"	<xflaim:ElementComponent " +
		"		xflaim:name=\"track\"> " +
		"		<xflaim:AttributeComponent " +
		"		  xflaim:name=\"index\" " +
		"		  xflaim:KeyComponent=\"2\" " +
		"		  xflaim:IndexOn=\"value\"/> " +
		"		<xflaim:AttributeComponent " +
		"		  xflaim:name=\"offset\" " +
		"		  xflaim:KeyComponent=\"3\" " +
		"		  xflaim:IndexOn=\"value\"/> " +
		"	</xflaim:ElementComponent> " +
		"</xflaim:Index> ";

		public bool importTests(
			Db			db,
			DbSystem	dbSystem)
		{
			DOMNode	doc = null;
			IStream	istream = null;

			// Create a document
			
			beginTest( "Import documents test");

			// Document #1

			try
			{
				istream = dbSystem.openBufferIStream( sDoc1);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling openBufferIStream - doc 1");
				return( false);
			}

			try
			{
				doc = db.importDocument( istream, (uint)PredefinedXFlaimCollections.XFLM_DATA_COLLECTION,
									doc, null);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling importDocument - doc 1");
				return( false);
			}

			// Document #2

			try
			{
				istream = dbSystem.openBufferIStream( sDoc2);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling openBufferIStream - doc 2");
				return( false);
			}

			try
			{
				doc = db.importDocument( istream, (uint)PredefinedXFlaimCollections.XFLM_DATA_COLLECTION,
					doc, null);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling importDocument - doc 2");
				return( false);
			}

			// Index definition

			try
			{
				istream = dbSystem.openBufferIStream( sIndexDef);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling openBufferIStream - index def");
				return( false);
			}

			try
			{
				doc = db.importDocument( istream, (uint)PredefinedXFlaimCollections.XFLM_DICT_COLLECTION,
					doc, null);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "calling importDocument - index def");
				return( false);
			}
			endTest( false, true);
			
			return( true);
		}

		public bool importTests(
			string	sDbName,
			DbSystem	dbSystem)
		{
			bool	bOk = false;
			Db		db = null;
			bool	bStartedTrans = false;
			RCODE	rc;

			// Create the database

			beginTest( "Create database \"" + sDbName + "\"");

			for (;;)
			{
				rc = RCODE.NE_XFLM_OK;
				try
				{
					XFLM_CREATE_OPTS	createOpts = new XFLM_CREATE_OPTS();

					createOpts.uiBlockSize = 8192;
					createOpts.uiVersionNum = (uint)DBVersions.XFLM_CURRENT_VERSION_NUM;
					createOpts.uiMinRflFileSize = 2000000;
					createOpts.uiMaxRflFileSize = 20000000;
					createOpts.bKeepRflFiles = 1;
					createOpts.bLogAbortedTransToRfl = 1;
					createOpts.eDefaultLanguage = Languages.FLM_DE_LANG;
					db = dbSystem.dbCreate( sDbName, null, null, null, null, createOpts);
				}
				catch (XFlaimException ex)
				{
					rc = ex.getRCode();

					if (rc != RCODE.NE_XFLM_FILE_EXISTS)
					{
						endTest( false, ex, "creating database");
						return( false);
					}
				}
				if (rc == RCODE.NE_XFLM_OK)
				{
					break;
				}

				// rc better be NE_XFLM_FILE_EXISTS - try to delete the file

				try
				{
					dbSystem.dbRemove( sDbName, null, null, true);
				}
				catch (XFlaimException ex)
				{
					endTest( false, ex, "removing database");
					return( false);
				}
			}
			endTest( false, true);

			// Start a transaction

			beginTest( "Start Update Transaction Test");
			try
			{
				db.transBegin( eDbTransType.XFLM_UPDATE_TRANS, 255, 0);
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "starting update transaction");
				goto Exit;
			}
			endTest( false, true);
			bStartedTrans = true;

			// Create a document

			if (!importTests( db, dbSystem))
			{
				goto Exit;
			}

			// Commit the transaction

			beginTest( "Commit Update Transaction Test");
			try
			{
				bStartedTrans = false;
				db.transCommit();
			}
			catch (XFlaimException ex)
			{
				endTest( false, ex, "committing update transaction");
				goto Exit;
			}
			endTest( false, true);

			bOk = true;

		Exit:
			if (bStartedTrans)
			{
				db.transAbort();
			}
			if (db != null)
			{
				db.close();
				db = null;
			}
			return( bOk);
		}
	}
}
