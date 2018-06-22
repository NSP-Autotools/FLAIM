//------------------------------------------------------------------------------
// Desc:	Sample program
//------------------------------------------------------------------------------

using System;
using System.IO;
using System.Runtime.InteropServices;
using xflaim;

namespace sample
{
	public class Sample
	{

		// Some global name ids to use when creating documents.

		// Namespace for all elements and attributes used in this sample program.

		public const string	sNamespace = "http://MyNamespace.com";

		// Element and attribute names

		public const string	sPersonElementName = "person";
		public const string	sUniqueIdAttributeName = "uniqueid";
		public const string	sNameElementName = "Name";
		public const string	sGivenElementName = "Given";
		public const string	sSurnameElementName = "Surname";
		public const string	sAddressElementName = "Address";
		public const string	sCityElementName = "City";
		public const string	sStateElementName = "State";
		public const string	sAgeElementName = "Age";
		public const string	sHomePhoneElementName = "HomePhone";
		public const string	sCellPhoneElementName = "CellPhone";

		// Name ids that correspond to the element and attribute names

		public uint				uiPersonElementId = 0;
		public uint				uiUniqueIdAttributeId = 0;
		public uint				uiNameElementId = 0;
		public uint				uiGivenElementId = 0;
		public uint				uiSurnameElementId = 0;
		public uint				uiAddressElementId = 0;
		public uint				uiCityElementId = 0;
		public uint				uiStateElementId = 0;
		public uint				uiAgeElementId = 0;
		public uint				uiHomePhoneElementId = 0;
		public uint				uiCellPhoneElementId = 0;

		//-----------------------------------------------------------------------
		// This method opens or creates the sample database.
		//-----------------------------------------------------------------------
		static Db createOrOpenDatabase( DbSystem dbSystem, out bool bCreatedDatabase)
		{
			Db			db;
			string	sDbName = "sample.db";

			// Try to open a database.  If that fails, create it.  The following
			// example creates the database in the current directory.  However,
			// a full or partial file name may be specified.
			// NOTE: Multiple threads should each do their own open of the
			// database and get back their own Db object.

			try
			{
				db = dbSystem.dbOpen( sDbName, null, null, null, false);
			}
			catch (XFlaimException ex)
			{
				if (ex.getRCode() != RCODE.NE_XFLM_IO_PATH_NOT_FOUND)
				{
					throw ex;
				}
				db = dbSystem.dbCreate( sDbName, null, null, null, null, null);
				bCreatedDatabase = true;
			}
			return( db);
		}

		//-----------------------------------------------------------------------
		// Create name Ids for elements and attributes in the dictionary.  If
		// we did NOT create the database, simply get the name IDs from the
		// dictionary.  The name IDs we need to create the document illustrated in
		// the createADocument method are as follows:
		//
		// 1. "person" element, namespace: "http://MyNamespace.com", data type: none
		// 2. "uniqueid" attributet, namespace: "http://MyNamespace.com", data type: number
		// 3. "Name" element, namespace: "http://MyNamespace.com", data type: none
		// 4. "Given" element, namespace: "http://MyNamespace.com", data type: text
		// 5. "Address" element, namespace: "http://MyNamespace.com", data type: none
		// 6. "City" element, namespace: "http://MyNamespace.com", data type: text
		// 7. "State" element, namespace: "http://MyNamespace.com", data type: text
		// 8. "Age" element, namespace: "http://MyNamespace.com", data type: number
		// 9. "HomePhone" element, namespace: "http://MyNamespace.com", data type: text
		// 10. "CellPhone" element, namespace: "http://MyNamespace.com", data type: text
		//-----------------------------------------------------------------------
		static void createOrGetNameIds( Db db, bool bCreatedDatabase)
		{
			if (bCreatedDatabase)
			{

				// Assume that the definitions need to be created in the dictionary.

				uiPersonElementId = db.createElementDef( sNamespace, sPersonElementName, FlmDataType.XFLM_NODATA_TYPE, 0);
				uiUniqueIdAttributeId = db.createAttributeDef( sNamespace, sUniqueIdAttributeName, FlmDataType.XFLM_NUMBER_TYPE, 0);
				uiNameElementId = db.createElementDef( sNamespace, sNameElementName, FlmDataType.XFLM_NODATA_TYPE, 0);
				uiGivenElementId = db.createElementDef( sNamespace, sGivenElementName, FlmDataType.XFLM_TEXT_TYPE, 0);
				uiSurnameElementId = db.createElementDef( sNamespace, sSurnameElementName, FlmDataType.XFLM_TEXT_TYPE, 0);
				uiAddressElementId = db.createElementDef( sNamespace, sAddressElementName, FlmDataType.XFLM_NODATA_TYPE, 0);
				uiCityElementId = db.createElementDef( sNamespace, sCityElementName, FlmDataType.XFLM_TEXT_TYPE, 0);
				uiStateElementId = db.createElementDef( sNamespace, sStateElementName, FlmDataType.XFLM_TEXT_TYPE, 0);
				uiAgeElementId = db.createElementDef( sNamespace, sAgeElementName, FlmDataType.XFLM_NUMBER_TYPE, 0);
				uiHomePhoneElementId = db.createElementDef( sNamespace, sHomePhoneElementName, FlmDataType.XFLM_TEXT_TYPE, 0);
				uiCellPhoneElementId = db.createElementDef( sNamespace, sCellPhoneElementName, FlmDataType.XFLM_TEXT_TYPE, 0);
			}
			else
			{

				// Assume that the definitions were created when we first created the database.

				uiPersonElementId = db.getElementNameId( sNamespace, sPersonElementName);
				uiUniqueIdAttributeId = db.getAttributeNameId( sNamespace, sUniqueIdAttributeName);
				uiNameElementId = db.getElementNameId( sNamespace, sNameElementName);
				uiGivenElementId = db.getElementNameId( sNamespace, sGivenElementName);
				uiSurnameElementId = db.getElementNameId( sNamespace, sSurnameElementName);
				uiAddressElementId = db.getElementNameId( sNamespace, sAddressElementName);
				uiCityElementId = db.getElementNameId( sNamespace, sCityElementName);
				uiStateElementId = db.getElementNameId( sNamespace, sStateElementName);
				uiAgeElementId = db.getElementNameId( sNamespace, sAgeElementName);
				uiHomePhoneElementId = db.getElementNameId( sNamespace, sHomePhoneElementName);
				uiCellPhoneElementId = db.getElementNameId( sNamespace, sCellPhoneElementName);
			}
		}

		//-----------------------------------------------------------------------
		// Create the following document:
		//
		// <myNamespace:person xmlns:myNamespace="http://MyNamespace.com" myNamespace:uniqueid="178442">
		//    <myNamespace:Name>
		//	      <myNamespace:Given>Peter</myNamespace:Given>
		//	      <myNamespace:Surname>Jones</myNamespace:Surname>
		//	   </myNamespace:Name>
		//    <myNamespace:Address>
		//	      <myNamespace:City>San Francisco</myNamespace:City>
		//	      <myNamespace:State>California</myNamespace:State>
		//	   </myNamespace:Address>
		//    <myNamespace:Age>37</myNamespace:Age>
		//    <myNamespace:HomePhone>307-888-1445</myNamespace:HomePhone>
		//    <myNamespace:CellPhone>307-612-1445</myNamespace:CellPhone>
		//	</myNamespace:person>
		//
		// It is assumed that the db object has already started an update transaction.
		//-----------------------------------------------------------------------
		static void createADocument( Db db)
		{
			DOMNode	rootNode = null;
			DOMNode	parentNode = null;
			DOMNode	childNode = null;
				
			// <myNamespace:person myNamespace:uniqueid="178442">
			rootNode = db.createRootElement( (uint)PredefinedXFlaimCollections.XFLM_DATA_COLLECTION, uiPersonElementId);
			rootNode.setAttributeValueULong( uiUniqueIdAttributeId, 178442);

			//    <myNamespace:Name>
			parentNode = rootNode.createNode( eDomNodeType.ELEMENT_NODE, uiNameElementId, eNodeInsertLoc.XFLM_FIRST_CHILD, parentNode);

			//	      <myNamespace:Given>Peter</myNamespace:Given>
			childNode = parentNode.createNode( eDomNodeType.ELEMENT_NODE, uiGivenElementId, eNodeInsertLoc.XFLM_FIRST_CHILD, childNode);
			childNode.setString( "Peter");

			//	      <myNamespace:Surname>Jones</myNamespace:Surname>
			childNode = parentNode.createNode( eDomNodeType.ELEMENT_NODE, uiSurnameElementId, eNodeInsertLoc.XFLM_LAST_CHILD, childNode);
			childNode.setString( "Jones");

			//    <myNamespace:Address>
			parentNode = rootNode.createNode( eDomNodeType.ELEMENT_NODE, uiAddressElementId, eNodeInsertLoc.XFLM_LAST_CHILD, parentNode);

			//	      <myNamespace:City>San Francisco</myNamespace:City>
			childNode = parentNode.createNode( eDomNodeType.ELEMENT_NODE, uiCityElementId, eNodeInsertLoc.XFLM_FIRST_CHILD, childNode);
			childNode.setString( "San Francisco");

			//	      <myNamespace:State>California</myNamespace:State>
			childNode = parentNode.createNode( eDomNodeType.ELEMENT_NODE, uiStateElementId, eNodeInsertLoc.XFLM_LAST_CHILD, childNode);
			childNode.setString( "California");

			//    <myNamespace:Age>37</myNamespace:Age>
			childNode = rootNode.createNode( eDomNodeType.ELEMENT_NODE, uiAgeElementId, eNodeInsertLoc.XFLM_LAST_CHILD, childNode);
			childNode.setUInt( 37);

			//    <myNamespace:HomePhone>307-888-1445</myNamespace:HomePhone>
			childNode = rootNode.createNode( eDomNodeType.ELEMENT_NODE, uiHomePhoneElementId, eNodeInsertLoc.XFLM_LAST_CHILD, childNode);
			childNode.setString( "307-888-1445");

			//    <myNamespace:CellPhone>307-612-1445</myNamespace:CellPhone>
			childNode = rootNode.createNode( eDomNodeType.ELEMENT_NODE, uiCellPhoneElementId, eNodeInsertLoc.XFLM_LAST_CHILD, childNode);
			childNode.setString( "307-612-1445");
		}

		//-----------------------------------------------------------------------
		// Find the document that has a given name of "Peter", state of "California", and
		// return the "Age" element from the document.  The XPATH query is as follows:
		//
		// person[Name/Given == "Peter" && Address/State == "California"]/Age
		//-----------------------------------------------------------------------
		static DOMNode queryDatabase( Db db)
		{
			Query	query = new Query( db, (uint)PredefinedXFlaimCollections.XFLM_DATA_COLLECTION);

			// First set up the query criteria.  This method calls the Query.addXXXXX methods to
			// set up the query.  An alternative way to set up the query criteria is to call the
			// Query.setupQueryExpr method, which is not illustrated here.

			// person
			query.addXPathComponent( eXPathAxisTypes.CHILD_AXIS, eDomNodeType.ELEMENT_NODE, uiPersonElementId);

			// [
			query.addOperator( eQueryOperators.XFLM_LBRACKET_OP, 0);

			// Name/Given
			query.addXPathComponent( eXPathAxisTypes.CHILD_AXIS, eDomNodeType.ELEMENT_NODE, uiNameElementId);
			query.addXPathComponent( eXPathAxisTypes.CHILD_AXIS, eDomNodeType.ELEMENT_NODE, uiGivenElementId);

			// == "Peter"
			query.addOperator( eQueryOperators.XFLM_EQ_OP, 0);
			query.addStringValue( "Peter");

			// &&
			query.addOperator( eQueryOperators.XFLM_AND_OP, 0);

			// Address/State
			query.addXPathComponent( eXPathAxisTypes.CHILD_AXIS, eDomNodeType.ELEMENT_NODE, uiAddressElementId);
			query.addXPathComponent( eXPathAxisTypes.CHILD_AXIS, eDomNodeType.ELEMENT_NODE, uiStateElementId);

			// == "California"
			query.addOperator( eQueryOperators.XFLM_EQ_OP, 0);
			query.addStringValue( "California");

			// ]
			query.addOperator( eQueryOperators.XFLM_RBRACKET_OP, 0);

			// /Age
			query.addXPathComponent( eXPathAxisTypes.CHILD_AXIS, eDomNodeType.ELEMENT_NODE, uiAgeElementId);


			// Retrieve and return the first result

			return( query.getFirst( null, 0));
		}

		//-----------------------------------------------------------------------
		// Main
		//-----------------------------------------------------------------------
		static void Main()
		{
			DbSystem dbSystem;
			Db			db;
			bool		bCreatedDatabase = false;

			// Must first get a DbSystem object in order to do anything with an
			// XFLAIM database.
				
			dbSystem = new DbSystem();

			// Create or open the database

			db = createOrOpenDatabase( dbSystem, out bCreatedDatabase);

			// Start an update transaction.  The timeout of 255 specifies that
			// the thread should wait forever to get the database lock.  A value
			// of zero would specify that it should not wait. In that case, or
			// for any value other than 255, the application should check to see
			// if the transaction failed to start because of a lock timeout
			// error. -- In general, the application will want to catch and
			// check for XFlaimException exceptions.  Such checking is not
			// shown in the following code.

			db.transBegin( eDbTransType.XFLM_UPDATE_TRANS, 255, 0);

			// Create a document (see createADocument above for illustration of the document as text)
			// In order to get the document, we must first get or create the needed element and
			// attribute name Ids in the dictionary.

			createOrGetNameIds( db, bCreatedDatabase);

			// Now create the document, as illustrated above.  The document is created in a collection that
			// is automatically created when the database is created,
			// the PredefinedXFlaimCollections.XFLM_DATA_COLLECTION

			createADocument( db);

			// Now we simply commit the transaction.  The document we created will be committed to
			// the database, and the database will be unlocked.

			db.transCommit();

			// Find the document that has a given name of "Peter", state of "California", and
			// return the "Age" element from the document.  The XPATH query is as follows:
			//
			// person[Name/Given == "Peter" && Address/State == "California"]/Age

			DOMNode	node = queryDatabase( db);

			// Output the age to the console.

			System.Console.WriteLine( "Age = {0}", node.getUInt());

			// Navigate to the cell phone element and print it out.

			for (;;)
			{
				node = node.getNextSibling( node);
				if (node.getNameId() == uiCellPhoneElementId)
				{
					System.Console.WriteLine( "Cell phone = {0}", node.getString());
					break;
				}
			}

		}
	}
}
