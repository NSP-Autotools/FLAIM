//------------------------------------------------------------------------------
// Desc:	Data Vector
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

package xflaim;

/**
 * This class implements an interface to the XFlaim IF_DataVector class.
 */
public class DataVector
{
	long		m_this;

	/**
	 * Constructor for the DataVector object.  This object provides access to
	 * the XFlaim IF_DataVector interface.  All methods defined by the
	 * IF_DataVector interace are accessible through this Java object.
	 * 
	 * @param lRef A reference to a C++ IF_DataVector object
	 */
	public DataVector(
		long		lRef)
	{
		super();
		m_this = lRef;
	}
	
	public long getThis()
	{
		return m_this;
	}
	
	/**
	 * Finalizer method, used to ensure that we release the actual C++ object.
	 */
	public void finalize()
	{
		if (m_this != 0)
		{
			_release( m_this);
			m_this = 0;
		}
	}
	
	/**
	 * Set the document ID of the data vector.
	 * @param lDocId Document ID for the data vector.
	 * @throws XFlaimException
	 */
	public void setDocumentID(
		long		lDocId) throws XFlaimException
	{
		_setDocumentId( m_this, lDocId);
	}
	
	/**
	 * Set the node ID of an element in the data vector. 
	 * @param iElementNumber Element whose node ID is to be set.
	 * @param lID Node ID value that is to be set.
	 * @throws XFlaimException
	 */
	public void setID(
		int			iElementNumber,
		long			lID) throws XFlaimException
	{
		_setID( m_this, iElementNumber, lID);
	}
	
	/**
	 * Set the name ID of an element in the data vector.
	 * @param iElementNumber Element whose name ID is to be set.
	 * @param iNameId Name ID that is to be set into the element.
	 * @param bIsAttr A boolean flag that indicates whether or not this
	 * name ID is an attribute or an element.  A value of true means it is
	 * an attribute, false means it is an element.
	 * @param bIsData A boolean flag that indicates whether or not the
	 * element is a data component as opposed to a key component.
	 * @throws XFlaimException
	 */
	public void setNameId(
		int			iElementNumber,
		int			iNameId,
		boolean		bIsAttr,
		boolean		bIsData) throws XFlaimException
	{
		_setNameId( m_this, iElementNumber, iNameId, bIsAttr, bIsData);
	}

	/**
	 * Set the value of an element in the data vector to a long value.
	 * @param iElementNumber Element whose value is to be set.
	 * @param lNum The long value that is to be set into the element.
	 * @throws XFlaimException
	 */
	public void setLong(
		int			iElementNumber,
		long			lNum) throws XFlaimException
	{
		_setLong( m_this, iElementNumber, lNum);
	}

	/**
	 * Set the value of an element in the data vector to a String value.
	 * @param iElementNumber Element whose value is to be set.
	 * @param sValue The string value that is to be set into the element.
	 * @throws XFlaimException
	 */
	public void setString(
		int			iElementNumber,
		String		sValue) throws XFlaimException
	{
		_setString( m_this, iElementNumber, sValue);
	}
	
	/**
	 * Set the value of an element in the data vector to a binary value.
	 * @param iElementNumber Element whose value is to be set.
	 * @param Value The binary value that is to be set into the element.
	 * @throws XFlaimException
	 */
	public void setBinary(
		int			iElementNumber,
		byte[]		Value) throws XFlaimException
	{
		_setBinary( m_this, iElementNumber, Value);
	}
	
	/**
	 * Set a flag of an element in the data vector to indicate that the value
	 * is right truncated.
	 * @param iElementNumber Element whose "right truncated" flag is to be set.
	 * @throws XFlaimException
	 */
	public void setRightTruncated(
		int		iElementNumber) throws XFlaimException
	{
		_setRightTruncated( m_this, iElementNumber);
	}

	/**
	 * Set a flag of an element in the data vector to indicate that the value
	 * is left truncated.
	 * @param iElementNumber Element whose "left truncated" flag is to be set.
	 * @throws XFlaimException
	 */
	public void setLeftTruncated(
		int		iElementNumber) throws XFlaimException
	{
		_setLeftTruncated( m_this, iElementNumber);
	}

	/**
	 * Clear the "right truncated" flag of an element in the data vector.
	 * @param iElementNumber Element whose "right truncated" flag is to be cleared.
	 * @throws XFlaimException
	 */
	public void clearRightTruncated(
		int		iElementNumber) throws XFlaimException
	{
		_clearRightTruncated( m_this, iElementNumber);
	}

	/**
	 * Clear the "left truncated" flag of an element in the data vector.
	 * @param iElementNumber Element whose "left truncated" flag is to be cleared.
	 * @throws XFlaimException
	 */
	public void clearLeftTruncated(
		int		iElementNumber) throws XFlaimException
	{
		_clearLeftTruncated( m_this, iElementNumber);
	}

	/**
	 * Determine if an element in the data vector is "right truncated."
	 * @param iElementNumber Element whose "right truncated" flag is to be checked.
	 * @return Returns flag indicating whether or not the specified element is
	 * "right truncated."
	 * @throws XFlaimException
	 */
	public void isRightTruncated(
		int		iElementNumber) throws XFlaimException
	{
		_isRightTruncated( m_this, iElementNumber);
	}

	/**
	 * Determine if an element in the data vector is "left truncated."
	 * @param iElementNumber Element whose "left truncated" flag is to be checked.
	 * @return Returns flag indicating whether or not the specified element is
	 * "left truncated."
	 * @throws XFlaimException
	 */
	public void isLeftTruncated(
		int		iElementNumber) throws XFlaimException
	{
		_isLeftTruncated( m_this, iElementNumber);
	}

	/**
	 * Get the Document ID for the data vector.
	 * @return Returns document ID.
	 * @throws XFlaimException
	 */
	public long getDocumentID() throws XFlaimException
	{
		return _getDocumentID( m_this);
	}

	/**
	 * Get the node ID for an element in the data vector.
	 * @param iElementNumber Element whose node ID is to be returned.
	 * @return Returns element's node ID.
	 * @throws XFlaimException
	 */
	public long getID(
		int		iElementNumber) throws XFlaimException
	{
		return _getID( m_this, iElementNumber);
	}

	/**
	 * Get the name ID for an element in the data vector.
	 * @param iElementNumber Element whose name ID is to be returned.
	 * @return Returns element's name ID.
	 * @throws XFlaimException
	 */
	public int getNameId(
		int		iElementNumber) throws XFlaimException
	{
		return _getNameId( m_this, iElementNumber);
	}

	/**
	 * Determine the name ID for an element in the data vector is an attribute
	 * name ID or an element name ID
	 * @param iElementNumber Element whose name ID is to be tested.
	 * @return Returns flag indicating if the name ID is an element name
	 * (returns false) or an attribute name (returns true).
	 * @throws XFlaimException
	 */
	public boolean isAttr(
		int		iElementNumber) throws XFlaimException
	{
		return _isAttr( m_this, iElementNumber);
	}

	/**
	 * Determine an element in the data vector is a data component.
	 * @param iElementNumber Element to be tested.
	 * @return Returns flag indicating if the element is a data component.
	 * @throws XFlaimException
	 */
	public boolean isDataComponent(
		int		iElementNumber) throws XFlaimException
	{
		return _isDataComponent( m_this, iElementNumber);
	}

	/**
	 * Determine an element in the data vector is a key component.
	 * @param iElementNumber Element to be tested.
	 * @return Returns flag indicating if the element is a key component.
	 * @throws XFlaimException
	 */
	public boolean isKeyComponent(
		int		iElementNumber) throws XFlaimException
	{
		return _isKeyComponent( m_this, iElementNumber);
	}

	/**
	 * Get the length of the data value of an element in the data vector.
	 * @param iElementNumber Element whose data value length is to be returned.
	 * @return Returns element's data value length.
	 * @throws XFlaimException
	 */
	public int getDataLength(
		int		iElementNumber) throws XFlaimException
	{
		return _getDataLength( m_this, iElementNumber);
	}

	/**
	 * Get the type of the data value of an element in the data vector.
	 * @param iElementNumber Element whose data value type is to be returned.
	 * @return Returns element's data value type.  This will be one of the
	 * values of {@link xflaim.FlmDataType FlmDataType}.
	 * @throws XFlaimException
	 */
	public int getDataType(
		int		iElementNumber) throws XFlaimException
	{
		return _getDataType( m_this, iElementNumber);
	}

	/**
	 * Get the value of an element in the data vector as a long value.
	 * @param iElementNumber Element whose data value is to be returned.
	 * @return Returns element's data value as a long.
	 * @throws XFlaimException
	 */
	public long getLong(
		int		iElementNumber) throws XFlaimException
	{
		return _getLong( m_this, iElementNumber);
	}
	
	/**
	 * Get the value of an element in the data vector as a String.
	 * @param iElementNumber Element whose data value is to be returned.
	 * @return Returns element's data value as a string.
	 * @throws XFlaimException
	 */
	public String getString(
		int		iElementNumber) throws XFlaimException
	{
		return _getString( m_this, iElementNumber);
	}

	/**
	 * Get the value of an element in the data vector as a binary value.
	 * @param iElementNumber Element whose data value is to be returned.
	 * @return Returns element's data value as a binary value.
	 * @throws XFlaimException
	 */
	public byte[] getBinary(
		int		iElementNumber) throws XFlaimException
	{
		return _getBinary( m_this, iElementNumber);
	}

	/**
	 * Return a buffer that is an index key for the elements in the vector.  The
	 * key is generated using the definition of the index from the specified
	 * database.
	 * @param jDb Database containing the index definition for which we want to
	 * generate a key.
	 * @param iIndexNum Index number for which we want to generate a key.
	 * @param bOutputIds Flag that specifies whether or not node ids and
	 * document ids in the data vector are to be included in the generated key.
	 * @return Returns the generated index key.
	 * @throws XFlaimException
	 */
	public byte[] outputKey(
		Db				jDb,
		int			iIndexNum,
		boolean		bOutputIds) throws XFlaimException
	{
		return _outputKey( m_this, jDb.m_this, iIndexNum, bOutputIds);
	}

	/**
	 * Return a buffer that contains the data component part of an index key.
	 * the buffer is composed from the elements in the data vector that are
	 * data components.
	 * @param jDb Database containing the index definition for which we want to
	 * generate the data component.
	 * @param iIndexNum Index number for which we want to generate the data
	 * component.
	 * @return Returns the generated data component part of an index key.
	 * @throws XFlaimException
	 */
	public byte[] outputData(
		Db			jDb,
		int		iIndexNum) throws XFlaimException
	{
		return _outputData( m_this, jDb.m_this, iIndexNum);
	}

	/**
	 * Populate the data vector's key components using a buffer that contains
	 * an index key.
	 * @param jDb Database containing the index definition that is to be used to
	 * parse through the key buffer to determine the key components.
	 * @param iIndexNum Index number to be used to parse through the key buffer
	 * to determine the key components.
	 * @param Key Buffer containing the index key to be parsed.
	 * @throws XFlaimException
	 */
	public void inputKey(
		Db				jDb,
		int			iIndexNum,
		byte[]		Key) throws XFlaimException
	{
		_inputKey( m_this, jDb.m_this, iIndexNum, Key);
	}

	/**
	 * Populate the data vector's data components using a buffer that contains
	 * an index key's data components.
	 * @param jDb Database containing the index definition that is to be used to
	 * parse through the data buffer to determine the data components.
	 * @param iIndexNum Index number to be used to parse through the data buffer
	 * to determine the data components.
	 * @param Data Buffer containing the index key's data components to be parsed.
	 * @throws XFlaimException
	 */
	public void inputData(
		Db				jDb,
		int			iIndexNum,
		byte[]		Data) throws XFlaimException
	{
		_inputData( m_this, jDb.m_this, iIndexNum, Data);
	}
	
	/**
	 * Reset the contents of the data vector.
	 * @throws XFlaimException
	 */
	public void reset() throws XFlaimException
	{
		_reset( m_this);
	}
	
// PRIVATE METHODS

	private native void _release( 
		long 		lThis);
	
	private native void _setDocumentId(
		long		lThis,
		long		lDocId) throws XFlaimException;

	private native void _setID(
		long		lThis,
		int		iElementNumber,
		long		lID) throws XFlaimException;
		
	private native void _setNameId(
		long		lThis,
		int		iElementNumber,
		int		iNameId,
		boolean	bIsAttr,
		boolean	bIsData) throws XFlaimException;
		
	private native void _setLong(
		long		lThis,
		int		iElementNumber,
		long		lNum) throws XFlaimException;
		
	private native void _setString(
		long		lThis,
		int		iElementNumber,
		String	sValue) throws XFlaimException;
	
	private native void _setBinary(
		long		lThis,
		int		iElementNumber,
		byte[]	Value) throws XFlaimException;
	
	private native void _setRightTruncated(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native void _setLeftTruncated(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native void _clearRightTruncated(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native void _clearLeftTruncated(
		long		lThis,
		int		iElementNumber) throws XFlaimException;
	
	private native boolean _isRightTruncated(
		long		lThis,
		int		iElementNumber) throws XFlaimException;
	
	private native boolean _isLeftTruncated(
		long		lThis,
		int		iElementNumber) throws XFlaimException;
	
	private native long _getDocumentID(
		long		lThis) throws XFlaimException;
	
	private native long _getID(
		long		lThis,
		int		iElementNumber) throws XFlaimException;
		
	private native int _getNameId(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native boolean _isAttr(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native boolean _isDataComponent(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native boolean _isKeyComponent(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native int _getDataLength(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native int _getDataType(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native long _getLong(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native String _getString(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native byte[] _getBinary(
		long		lThis,
		int		iElementNumber) throws XFlaimException;

	private native byte[] _outputKey(
		long		lThis,
		long		ljDbRef,
		int		iIndexNum,
		boolean	bOutputIds) throws XFlaimException;

	private native byte[] _outputData(
		long		lThis,
		long		ljDbRef,
		int		iIndexNum) throws XFlaimException;

	private native void _inputKey(
		long		lThis,
		long		ljDbRef,
		int		iIndexNum,
		byte[]	Key) throws XFlaimException;

	private native void _inputData(
		long		lThis,
		long		ljDbRef,
		int		iIndexNum,
		byte[]	Data) throws XFlaimException;

	private native void _reset(
		long		lThis) throws XFlaimException;
}

