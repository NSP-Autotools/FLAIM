//------------------------------------------------------------------------------
// Desc:	Data Vector Class
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
using System.Runtime.InteropServices;

namespace xflaim
{

	/// <summary>
	/// The DataVector class provides a number of methods that allow C#
	/// applications to access data vector object in unmanaged space.  A
	/// DataVector object is obtained by calling <see cref="DbSystem.createDataVector"/>.
	/// </summary>
	public class DataVector
	{
		private IntPtr 	m_pDataVector;			// Pointer to IF_DataVector object in unmanaged space
		private DbSystem 	m_dbSystem;

		/// <summary>
		/// Db constructor.
		/// </summary>
		/// <param name="pDataVector">
		/// Reference to an IF_Db object.
		/// </param>
		/// <param name="dbSystem">
		/// DbSystem object that this Db object is associated with.
		/// </param>
		internal DataVector(
			IntPtr	pDataVector,
			DbSystem	dbSystem)
		{
			if (pDataVector == IntPtr.Zero)
			{
				throw new XFlaimException( "Invalid IF_DataVector reference");
			}
			
			m_pDataVector = pDataVector;

			if (dbSystem == null)
			{
				throw new XFlaimException( "Invalid DbSystem reference");
			}
			
			m_dbSystem = dbSystem;
			
			// Must call something inside of DbSystem.  Otherwise, the
			// m_dbSystem object gets a compiler warning on linux because
			// it is not used anywhere.  Other than that, there is really
			// no need to make the following call.
			if (m_dbSystem.getDbSystem() == IntPtr.Zero)
			{
				throw new XFlaimException( "Invalid DbSystem.IF_DbSystem object");
			}
		}

		/// <summary>
		/// Destructor.
		/// </summary>
		~DataVector()
		{
			close();
		}

		/// <summary>
		/// Return the pointer to the IF_DataVector object.
		/// </summary>
		/// <returns>Returns a pointer to the IF_DataVector object.</returns>
		internal IntPtr getDataVector()
		{
			return( m_pDataVector);
		}

		/// <summary>
		/// Close this data vector object.
		/// </summary>
		public void close()
		{
			// Release the native pDataVector!
		
			if (m_pDataVector != IntPtr.Zero)
			{
				xflaim_DataVector_Release( m_pDataVector);
				m_pDataVector = IntPtr.Zero;
			}
		
			// Remove our reference to the dbSystem so it can be released.
		
			m_dbSystem = null;
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DataVector_Release(
			IntPtr	pDataVector);

//-----------------------------------------------------------------------------
// setDocumentID
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set the document ID a data vector is associated with.
		/// </summary>
		/// <param name="ulDocId">
		/// Docment ID.
		/// </param>
		public void setDocumentID(
			ulong		ulDocId)
		{
			xflaim_DataVector_setDocumentID( m_pDataVector, ulDocId);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DataVector_setDocumentID(
			IntPtr		pDataVector,
			ulong			ulDocId);

//-----------------------------------------------------------------------------
// setID
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set the ID for a particular element in a data vector.
		/// </summary>
		/// <param name="uiElementNumber">
		/// Element number whose ID is to be set.
		/// </param>
		/// <param name="ulID">
		/// ID to be set for the element.  This is generally a node ID.
		/// </param>
		public void setID(
			uint		uiElementNumber,
			ulong		ulID)
		{
			RCODE	rc;
			if ((rc = xflaim_DataVector_setID( m_pDataVector, uiElementNumber, ulID)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_setID(
			IntPtr		pDataVector,
			uint			uiElementNumber,
			ulong			ulID);

//-----------------------------------------------------------------------------
// setNameId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set the name ID for the specified element in the data vector.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be set</param>
		/// <param name="uiNameId">Name Id to be set</param>
		/// <param name="bIsAttr">Flag that specifies whether the value is an attribute</param>
		/// <param name="bIsData">Flag that specifies whether the value is a data component</param>
		public void setNameId(
			uint			uiElementNumber,
			uint			uiNameId,
			bool			bIsAttr,
			bool			bIsData)
		{
			RCODE	rc;

			if ((rc = xflaim_DataVector_setNameId( m_pDataVector,
				uiElementNumber, uiNameId, (int)(bIsAttr ? 1 : 0),
				(int)(bIsData ? 1 : 0))) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_setNameId(
			IntPtr		pDataVector,
			uint			uiElementNumber,
			uint			uiNameId,
			int			bIsAttr,
			int			bIsData);

//-----------------------------------------------------------------------------
// setULong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set an unsigned long value for the specified element in the data vector.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be set</param>
		/// <param name="ulValue">Value to be set</param>
		public void setULong(
			uint			uiElementNumber,
			ulong			ulValue)
		{
			RCODE	rc;

			if ((rc = xflaim_DataVector_setULong( m_pDataVector,
				uiElementNumber, ulValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_setULong(
			IntPtr		pDataVector,
			uint			uiElementNumber,
			ulong			ulValue);

//-----------------------------------------------------------------------------
// setLong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set a long value for the specified element in the data vector.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be set</param>
		/// <param name="lValue">Value to be set</param>
		public void setLong(
			uint			uiElementNumber,
			long			lValue)
		{
			RCODE	rc;

			if ((rc = xflaim_DataVector_setLong( m_pDataVector,
				uiElementNumber, lValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_setLong(
			IntPtr		pDataVector,
			uint			uiElementNumber,
			long			lValue);

//-----------------------------------------------------------------------------
// setUInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set an unsigned int value for the specified element in the data vector.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be set</param>
		/// <param name="uiValue">Value to be set</param>
		public void setUInt(
			uint			uiElementNumber,
			uint			uiValue)
		{
			RCODE	rc;

			if ((rc = xflaim_DataVector_setUInt( m_pDataVector,
				uiElementNumber, uiValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_setUInt(
			IntPtr		pDataVector,
			uint			uiElementNumber,
			uint			uiValue);

//-----------------------------------------------------------------------------
// setInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set an int value for the specified element in the data vector.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be set</param>
		/// <param name="iValue">Value to be set</param>
		public void setInt(
			uint			uiElementNumber,
			int			iValue)
		{
			RCODE	rc;

			if ((rc = xflaim_DataVector_setInt( m_pDataVector,
				uiElementNumber, iValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_setInt(
			IntPtr		pDataVector,
			uint			uiElementNumber,
			int			iValue);

//-----------------------------------------------------------------------------
// setString
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set a string value for the specified element in the data vector.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be set</param>
		/// <param name="sValue">Value to be set</param>
		public void setString(
			uint			uiElementNumber,
			string		sValue)
		{
			RCODE	rc;

			if ((rc = xflaim_DataVector_setString( m_pDataVector,
				uiElementNumber, sValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_setString(
			IntPtr		pDataVector,
			uint			uiElementNumber,
			[MarshalAs(UnmanagedType.LPWStr)]
			string		sValue);

//-----------------------------------------------------------------------------
// setBinary
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set a binary value for the specified element in the data vector.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be set</param>
		/// <param name="pvValue">Value to be set</param>
		public void setBinary(
			uint			uiElementNumber,
			byte []		pvValue)
		{
			RCODE	rc;

			if ((rc = xflaim_DataVector_setBinary( m_pDataVector,
				uiElementNumber, pvValue, (uint)pvValue.Length)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_setBinary(
			IntPtr			pDataVector,
			uint				uiElementNumber,
			[MarshalAs(UnmanagedType.LPArray), In]
			byte []			pvValue,
			uint				uiLen);

//-----------------------------------------------------------------------------
// setRightTruncated
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set the right truncated flag for the specified element in the vector.
		/// </summary>
		/// <param name="uiElementNumber">Element whose right truncation flag is to be set</param>
		public void setRightTruncated(
			uint		uiElementNumber)
		{
			xflaim_DataVector_setRightTruncated( m_pDataVector, uiElementNumber);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DataVector_setRightTruncated(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// setLeftTruncated
//-----------------------------------------------------------------------------

		/// <summary>
		/// Set the left truncated flag for the specified element in the vector.
		/// </summary>
		/// <param name="uiElementNumber">Element whose left truncation flag is to be set</param>
		public void setLeftTruncated(
			uint		uiElementNumber)
		{
			xflaim_DataVector_setLeftTruncated( m_pDataVector, uiElementNumber);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DataVector_setLeftTruncated(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// clearRightTruncated
//-----------------------------------------------------------------------------

		/// <summary>
		/// Clear the right truncated flag for the specified element in the vector.
		/// </summary>
		/// <param name="uiElementNumber">Element whose right truncation flag is to be cleared</param>
		public void clearRightTruncated(
			uint		uiElementNumber)
		{
			xflaim_DataVector_clearRightTruncated( m_pDataVector, uiElementNumber);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DataVector_clearRightTruncated(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// clearLeftTruncated
//-----------------------------------------------------------------------------

		/// <summary>
		/// Clear the left truncated flag for the specified element in the vector.
		/// </summary>
		/// <param name="uiElementNumber">Element whose left truncation flag is to be cleared</param>
		public void clearLeftTruncated(
			uint		uiElementNumber)
		{
			xflaim_DataVector_clearLeftTruncated( m_pDataVector, uiElementNumber);
		}

		[DllImport("xflaim")]
		private static extern void xflaim_DataVector_clearLeftTruncated(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// isRightTruncated
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine if the right truncated flag for the specified element in the vector is set.
		/// </summary>
		/// <param name="uiElementNumber">Element to be tested</param>
		/// <returns>
		/// Returns a flag indicating whether the specified element's right truncation flag is set.
		/// </returns>
		public bool isRightTruncated(
			uint		uiElementNumber)
		{
			return( xflaim_DataVector_isRightTruncated( m_pDataVector, uiElementNumber) != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern int xflaim_DataVector_isRightTruncated(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// isLeftTruncated
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine if the left truncated flag for the specified element in the vector is set.
		/// </summary>
		/// <param name="uiElementNumber">Element to be tested</param>
		/// <returns>
		/// Returns a flag indicating whether the specified element's left truncation flag is set.
		/// </returns>
		public bool isLeftTruncated(
			uint		uiElementNumber)
		{
			return( xflaim_DataVector_isLeftTruncated( m_pDataVector, uiElementNumber) != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern int xflaim_DataVector_isLeftTruncated(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// getDocumentID
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the document ID associated with the data vector.
		/// </summary>
		/// <returns>
		/// Document ID for the data vector.
		/// </returns>
		public ulong getDocumentID()
		{
			return( xflaim_DataVector_getDocumentID( m_pDataVector));
		}

		[DllImport("xflaim")]
		private static extern ulong xflaim_DataVector_getDocumentID(
			IntPtr			pDataVector);

//-----------------------------------------------------------------------------
// getID
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the ID associated with the specified element in the data vector.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose ID is to be returned.</param>
		/// <returns>
		/// ID for the specified element in the data vector.
		/// </returns>
		public ulong getID(
			uint		uiElementNumber)
		{
			return( xflaim_DataVector_getID( m_pDataVector, uiElementNumber));
		}

		[DllImport("xflaim")]
		private static extern ulong xflaim_DataVector_getID(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// getNameId
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the name ID associated with the specified element in the data vector.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose name ID is to be returned.</param>
		/// <returns>
		/// Name ID for the specified element in the data vector.
		/// </returns>
		public uint getNameId(
			uint		uiElementNumber)
		{
			return( xflaim_DataVector_getNameId( m_pDataVector, uiElementNumber));
		}

		[DllImport("xflaim")]
		private static extern uint xflaim_DataVector_getNameId(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// isAttr
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine if the specified element in the data vector is an attribute.
		/// </summary>
		/// <param name="uiElementNumber">Element number to be tested.</param>
		/// <returns>
		/// Flag indicating whether the specified element in the data vector is an attribute.
		/// </returns>
		public bool isAttr(
			uint		uiElementNumber)
		{
			return( xflaim_DataVector_isAttr( m_pDataVector, uiElementNumber) != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern int xflaim_DataVector_isAttr(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// isDataComponent
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine if the specified element in the data vector is a data component.
		/// </summary>
		/// <param name="uiElementNumber">Element number to be tested.</param>
		/// <returns>
		/// Flag indicating whether the specified element in the data vector is a data component.
		/// </returns>
		public bool isDataComponent(
			uint		uiElementNumber)
		{
			return( xflaim_DataVector_isDataComponent( m_pDataVector, uiElementNumber) != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern int xflaim_DataVector_isDataComponent(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// isKeyComponent
//-----------------------------------------------------------------------------

		/// <summary>
		/// Determine if the specified element in the data vector is a key component.
		/// </summary>
		/// <param name="uiElementNumber">Element number to be tested.</param>
		/// <returns>
		/// Flag indicating whether the specified element in the data vector is a key component.
		/// </returns>
		public bool isKeyComponent(
			uint		uiElementNumber)
		{
			return( xflaim_DataVector_isKeyComponent( m_pDataVector, uiElementNumber) != 0 ? true : false);
		}

		[DllImport("xflaim")]
		private static extern int xflaim_DataVector_isKeyComponent(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// getDataLength
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the length of the data that is stored in the specified element in the data vector.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose data length is to be returned.</param>
		/// <returns>
		/// Length of data in the specified element.
		/// </returns>
		public uint getDataLength(
			uint		uiElementNumber)
		{
			return( xflaim_DataVector_getDataLength( m_pDataVector, uiElementNumber));
		}

		[DllImport("xflaim")]
		private static extern uint xflaim_DataVector_getDataLength(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// getDataType
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the type of the data that is stored in the specified element in the data vector.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose data type is to be returned.</param>
		/// <returns>
		/// Type of data in the specified element.
		/// </returns>
		public FlmDataType getDataType(
			uint		uiElementNumber)
		{
			return( xflaim_DataVector_getDataType( m_pDataVector, uiElementNumber));
		}

		[DllImport("xflaim")]
		private static extern FlmDataType xflaim_DataVector_getDataType(
			IntPtr			pDataVector,
			uint				uiElementNumber);

//-----------------------------------------------------------------------------
// getULong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the data in the specified element in the data vector as an unsigned
		/// long value.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be returned.</param>
		/// <returns>
		/// Element's value is returned as an unsigned long.
		/// </returns>
		public ulong getULong(
			uint		uiElementNumber)
		{
			RCODE	rc;
			ulong	ulValue;

			if ((rc = xflaim_DataVector_getULong( m_pDataVector, uiElementNumber, out ulValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ulValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_getULong(
			IntPtr			pDataVector,
			uint				uiElementNumber,
			out ulong		pulValue);

//-----------------------------------------------------------------------------
// getLong
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the data in the specified element in the data vector as a
		/// long value.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be returned.</param>
		/// <returns>
		/// Element's value is returned as a long.
		/// </returns>
		public long getLong(
			uint		uiElementNumber)
		{
			RCODE	rc;
			long	lValue;

			if ((rc = xflaim_DataVector_getLong( m_pDataVector, uiElementNumber, out lValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( lValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_getLong(
			IntPtr			pDataVector,
			uint				uiElementNumber,
			out long			plValue);

//-----------------------------------------------------------------------------
// getUInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the data in the specified element in the data vector as an unsigned
		/// int value.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be returned.</param>
		/// <returns>
		/// Element's value is returned as an unsigned int.
		/// </returns>
		public uint getUInt(
			uint		uiElementNumber)
		{
			RCODE	rc;
			uint	uiValue;

			if ((rc = xflaim_DataVector_getUInt( m_pDataVector, uiElementNumber, out uiValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( uiValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_getUInt(
			IntPtr			pDataVector,
			uint				uiElementNumber,
			out uint			puiValue);

//-----------------------------------------------------------------------------
// getInt
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the data in the specified element in the data vector as a signed
		/// int value.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be returned.</param>
		/// <returns>
		/// Element's value is returned as a signed int.
		/// </returns>
		public int getInt(
			uint		uiElementNumber)
		{
			RCODE	rc;
			int	iValue;

			if ((rc = xflaim_DataVector_getInt( m_pDataVector, uiElementNumber, out iValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( iValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_getInt(
			IntPtr			pDataVector,
			uint				uiElementNumber,
			out int			piValue);

//-----------------------------------------------------------------------------
// getString
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the data in the specified element in the data vector as a string
		/// value.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be returned.</param>
		/// <returns>
		/// Element's value is returned as a string.
		/// </returns>
		public string getString(
			uint		uiElementNumber)
		{
			RCODE		rc;
			IntPtr	pValue;
			string	sValue;

			if ((rc = xflaim_DataVector_getString( m_pDataVector, uiElementNumber, out pValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			sValue = Marshal.PtrToStringUni( pValue);
			m_dbSystem.freeUnmanagedMem( pValue);
			return( sValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_getString(
			IntPtr			pDataVector,
			uint				uiElementNumber,
			out IntPtr		ppValue);

//-----------------------------------------------------------------------------
// getBinary
//-----------------------------------------------------------------------------

		/// <summary>
		/// Get the data in the specified element in the data vector as a byte array
		/// value.
		/// </summary>
		/// <param name="uiElementNumber">Element number whose value is to be returned.</param>
		/// <returns>
		/// Element's value is returned as a byte array.
		/// </returns>
		public byte [] getBinary(
			uint		uiElementNumber)
		{
			RCODE		rc;
			byte []	ucValue = null;
			uint		uiLen = xflaim_DataVector_getDataLength( m_pDataVector, uiElementNumber);

			ucValue = new byte [uiLen];

			if ((rc = xflaim_DataVector_getBinary( m_pDataVector, uiElementNumber,
								uiLen, ucValue)) != 0)
			{
				throw new XFlaimException( rc);
			}
			return( ucValue);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_getBinary(
			IntPtr			pDataVector,
			uint				uiElementNumber,
			uint				uiLen,
			[MarshalAs(UnmanagedType.LPArray), Out] 
			byte []			pvValue);

//-----------------------------------------------------------------------------
// outputKey
//-----------------------------------------------------------------------------

		/// <summary>
		/// Return a buffer that is an index key for the elements in the vector.  The
		/// key is generated using the definition of the index from the specified
		/// database.
		/// </summary>
		/// <param name="db">
		/// Database containing the index definition for which we want to
		/// generate a key.
		/// </param>
		/// <param name="uiIndexNum">Index number for which we want to generate a key.</param>
		/// <param name="bOutputIds">
		/// Flag that specifies whether or not node ids and
		/// document ids in the data vector are to be included in the generated key.
		/// </param>
		/// <returns>
		/// Returns the generated index key.
		/// </returns>
		public byte[] outputKey(
			Db				db,
			uint			uiIndexNum,
			bool			bOutputIds)
		{
			RCODE		rc;
			byte []	ucKey = new byte [DbSystem.XFLM_MAX_KEY_SIZE];
			int		iLen;
			byte []	ucReturnKey;

			if ((rc = xflaim_DataVector_outputKey( m_pDataVector, db.getDb(), uiIndexNum,
				(int)(bOutputIds ? 1 : 0), ucKey, out iLen)) != 0)
			{
				throw new XFlaimException( rc);
			}
			ucReturnKey = new byte [iLen];
			Array.Copy( ucKey, ucReturnKey, iLen);
			return( ucReturnKey);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_outputKey(
			IntPtr			pDataVector,
			IntPtr			pDb,
			uint				uiIndexNum,
			int				bOutputIds,
			[MarshalAs(UnmanagedType.LPArray), Out] 
			byte []			pucKey,
			out int			piLen);

//-----------------------------------------------------------------------------
// outputData
//-----------------------------------------------------------------------------

		/// <summary>
		/// Return a buffer that contains the data components for an index key
		/// for the elements in the vector.  The data is generated using the
		/// definition of the index from the specified database.
		/// </summary>
		/// <param name="db">
		/// Database containing the index definition for which we want to
		/// generate a key.
		/// </param>
		/// <param name="uiIndexNum">Index number for which we want to generate a key.</param>
		/// <returns>
		/// Returns the data components of the index key.
		/// </returns>
		public byte[] outputData(
			Db				db,
			uint			uiIndexNum)
		{
			RCODE		rc;
			int		iBufSize = 512;
			byte []	ucData = new byte [iBufSize];
			int		iLen;
			byte []	ucReturnData;

			// Try until we have a large enough buffer.

			for (;;)
			{
				if ((rc = xflaim_DataVector_outputData( m_pDataVector, db.getDb(), uiIndexNum,
					ucData, iBufSize, out iLen)) != 0)
				{
					if (rc != RCODE.NE_XFLM_CONV_DEST_OVERFLOW)
					{
						throw new XFlaimException( rc);
					}
					else
					{
						iBufSize += 2048;
						ucData = new byte [iBufSize];
					}
				}
				else
				{
					break;
				}
			}
			ucReturnData = new byte [iLen];
			Array.Copy( ucData, ucReturnData, iLen);
			return( ucReturnData);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_outputData(
			IntPtr			pDataVector,
			IntPtr			pDb,
			uint				uiIndexNum,
			[MarshalAs(UnmanagedType.LPArray), Out] 
			byte []			pucData,
			int				iBufSize,
			out int			piLen);

//-----------------------------------------------------------------------------
// inputKey
//-----------------------------------------------------------------------------

		/// <summary>
		/// Populate the data vector's key components using a buffer that contains
		/// an index key's key components.
		/// </summary>
		/// <param name="db">
		/// Database containing the index definition that is to be used to
		/// parse through the key buffer to determine the key components.
		/// </param>
		/// <param name="uiIndexNum">
		/// Index number to be used to parse through the key buffer
		/// to determine the key components.
		/// </param>
		/// <param name="ucKey">
		/// Buffer containing the index key to be parsed.
		/// </param>
		public void inputKey(
			Db				db,
			uint			uiIndexNum,
			byte[]		ucKey)
		{
			RCODE		rc;

			if ((rc = xflaim_DataVector_inputKey( m_pDataVector, db.getDb(), uiIndexNum,
					ucKey, ucKey.Length)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_inputKey(
			IntPtr			pDataVector,
			IntPtr			pDb,
			uint				uiIndexNum,
			[MarshalAs(UnmanagedType.LPArray), In] 
			byte []			pucKey,
			int				iKeyLen);

//-----------------------------------------------------------------------------
// inputData
//-----------------------------------------------------------------------------

		/// <summary>
		/// Populate the data vector's data components using a buffer that contains
		/// an index key's data components.
		/// </summary>
		/// <param name="db">
		/// Database containing the index definition that is to be used to
		/// parse through the key buffer to determine the data components.
		/// </param>
		/// <param name="uiIndexNum">
		/// Index number to be used to parse through the data buffer
		/// to determine the data components.
		/// </param>
		/// <param name="ucData">
		/// Buffer containing the data to be parsed.
		/// </param>
		public void inputData(
			Db				db,
			uint			uiIndexNum,
			byte[]		ucData)
		{
			RCODE		rc;

			if ((rc = xflaim_DataVector_inputData( m_pDataVector, db.getDb(), uiIndexNum,
				ucData, ucData.Length)) != 0)
			{
				throw new XFlaimException( rc);
			}
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_inputData(
			IntPtr			pDataVector,
			IntPtr			pDb,
			uint				uiIndexNum,
			[MarshalAs(UnmanagedType.LPArray), In] 
			byte []			pucData,
			int				iDataLen);

//-----------------------------------------------------------------------------
// reset
//-----------------------------------------------------------------------------

		/// <summary>
		/// Reset the contents of a data vector.
		/// </summary>
		public void reset()
		{
			xflaim_DataVector_reset( m_pDataVector);
		}

		[DllImport("xflaim")]
		private static extern RCODE xflaim_DataVector_reset(
			IntPtr	pDataVector);

	}
}
