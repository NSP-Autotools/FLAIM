//------------------------------------------------------------------------------
// Desc:	Db System
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
 * The DbSystem class provides a number of methods that allow java 
 * applications to access the XFlaim native environment, specifically, the 
 * IF_DbSystem interface.
 */
public class DbSystem
{
	static
	{ 
		System.loadLibrary( "xflaim");
	}
	  
	/**
	 * Loads the appropriate native library (as determined from the system
	 * properties).
	 * 
	 * @throws XFlaimException
	 */
	public DbSystem()
			throws XFlaimException
	{
		super();
		m_this = _createDbSystem();
	}

	public void finalize()
	{
		_release( m_this);
		m_this = 0;
	}	

	public void dbClose()
	{
		_release( m_this);
		m_this = 0;		
	}
		
	/**
	 * Creates a new XFlaim database.
	 * 
	 * @param sDbFileName This is the name of the control file for the database.
	 * The control file is the primary database name.  It may include a full
	 * or partial directory name, or no directory name.  If a partial directory
	 * name or is included, it is assumed to be relative to the current working
	 * directory.  If no directory is specified, the file will be created in
	 * the current working directory.
	 * @param sDataDir The directory where the database data files are stored.
	 * If null, the data files will be stored in the same directory as the control
	 * file.
	 * @param sRflDir The directory where the roll forward log files should be
	 * stored.  If null, this defaults to the same directory where the control file
	 * exists.  Within this directory, XFLAIM expects a subdirectory to exist that
	 * holds the RFL files.  The subdirectory name is derived from the control
	 * file's base name. If the control file's base name has an extension
	 * of ".db", the ".db" is replaced with ".rfl".  If the control file's base
	 * name does not have an extension of ".db", an extension of ".rfl" is simply
	 * appended to the control file's base name.  For example, if the control file's
	 * base name is "MyDatabase.db", the subdirectory will be named "MyDatabase.rfl".
	 * If the control file's base name is "MyDatabase.xyz", the subdirectory will be
	 * named "MyDatabase.xyz.rfl".
	 * @param sDictFileName - The name of a file which contains dictionary
	 * definition items.  May be null.  Ignored if sDictBuf is non-null.
	 * @param sDictBuf - Contains dictionary definitions.  If null,
	 * sDictFileName is used.  If both sDictFileName and sDictBuf are null,
	 * the database is created with an empty dictionary.
	 * @param createOpts - A {@link xflaim.CREATEOPTS CREATEOPTS} object that
	 * contains several parameters that affect
	 * the creation of the database.  (For advanced users.) 
	 * @return Returns an instance of a {@link xflaim.Db Db} object.
	 * @throws XFlaimException
	 */
	public Db dbCreate(
		String 		sDbFileName,
		String 		sDataDir,
		String 		sRflDir,
		String 		sDictFileName,
		String 		sDictBuf,
		CREATEOPTS  createOpts) throws XFlaimException
	{
	
		Db 		jDb = null;
		long 		jDb_ref;
		
		jDb_ref = _dbCreate( m_this, sDbFileName, sDataDir, sRflDir,
						sDictFileName, sDictBuf, createOpts);
		
		if( jDb_ref != 0)
		{
			jDb = new Db( jDb_ref, this);	
		}
		
		return( jDb);
	}
	
	/**
	 * Opens an existing XFlaim database.
	 * @param sDbFileName The name of the control file of the database to open.
	 * For more explanation see documentation for dbCreate.
	 * @param sDataDir The data file directory.  See dbCreate for more
	 * information.
	 * @param sRflDir The roll-forward log file directory.  See dbCreate for more
	 * information.
	 * @param sPassword Password for opening the database.  This is only needed
	 * if the database key is currently wrapped in a password instead of the
	 * local NICI storage key.
	 * @return Returns an instance of a {@link xflaim.Db Db} object.
	 * @throws XFlaimException
	 */
	 
	public Db dbOpen(
		String	sDbFileName,
		String	sDataDir,
		String	sRflDir,
		String	sPassword,
		boolean	bAllowLimited) throws XFlaimException
	{
		Db 	jDb = null;
		long 	jDb_ref;
											
		if( (jDb_ref = _dbOpen( m_this, sDbFileName, sDataDir, 
			sRflDir, sPassword, bAllowLimited)) != 0)
		{
			jDb = new Db( jDb_ref, this);
		}
		
		return( jDb);
	}
	
	/**
	 * Removes (deletes) an XFlaim database.
	 * @param sDbFileName The name of the control file of the database to delete.
	 * For more information see dbCreate.
	 * @param sDataDir The data file directory.  See dbCreate for more
	 * information.
	 * @param sRflDir The roll-forward log file directory.  See dbCreate for more
	 * information.
	 * @param bRemoveRflFiles If true, the roll forward log files will be
	 * deleted.
	 */
	public void dbRemove(
		String	sDbFileName,
		String	sDataDir,
		String	sRflDir,
		boolean	bRemoveRflFiles) throws XFlaimException
	{
		_dbRemove( m_this, sDbFileName, sDataDir, sRflDir, bRemoveRflFiles);
	}
	
	/**
	 * Restores a previously backed up database.  <code>sBackupPath</code> and 
	 * <code> RestoreClient</code> are mutually exclusive.  If
	 * <code>RestoreClient</code> is null, then an instance of
	 * <code>DefaultRestoreClient</code> will be created and
	 * <code>sBackupPath</code> passed into its constructor.  If <code>
	 * RestoreClient</code> is non-null, <code>sBackupPath</code> is ignored.
	 * @param sDbPath The name of the control file of the database to restore.
	 * @param sDataDir The data file directory.  See dbCreate for more
	 * information.
	 * @param sRflDir The roll-forward log file directory.  See dbCreate for more
	 * information.
	 * @param sBackupPath The path to the backup files.  This may be null.  If
	 * non-null, it specifies the directory where the backup files which are
	 * to be restored are found.  If null, the restoreClient parameter must be
	 * non-null.
	 * @param sPassword Password for the backup.  If non-null, the database key in
	 * the backup was wrapped in a password instead of the local NICI storage
	 * key.  This allows the database to be restored to a different machine if
	 * desired.  If null, the database can only be restored to the same machine
	 * where it originally existed.
	 * @param restoreClient An object implementing the
	 * {@link RestoreClient RestoreClient} interface.  This may be null.  If
	 * non-null, it is an object that knows how to get the backup data.
	 * @param restoreStatus An object implementing the
	 * {@link RestoreStatus RestoreStatus} interface.  This may be null.  If
	 * non-null, it is a callback object whose methods will be called to report
	 * restore progress.
	 * @throws XFlaimException
	 */
	public void dbRestore(
		String			sDbPath,
		String			sDataDir,
		String			sRflDir,
		String			sBackupPath,
		String			sPassword,
		RestoreClient	restoreClient,
		RestoreStatus	restoreStatus) throws XFlaimException
	{
		RestoreClient	client;
		
		if (restoreClient != null)
		{
			client = restoreClient;
		}
		else
		{
			client = new DefaultRestoreClient( sBackupPath);
		}
		
		_dbRestore( m_this, sDbPath, sDataDir, sRflDir, sBackupPath,
				sPassword, client, restoreStatus);
	}

	/**
	 * Peforms an integrity check on the specified database.
	 * @param sDbFileName The name of the control file of the database to be checked.
	 * @param sDataDir The data file directory.  See dbCreate for more
	 * information.
	 * @param sRflDir The roll-forward log file directory.  See dbCreate for more
	 * information.
	 * @param sPassword Password for opening the database.  This is only needed
	 * if the database key is currently wrapped in a password instead of the
	 * local NICI storage key.
	 * @param iFlags Flags that control exactly what the operation checks.
	 * Should be a logical OR of the members of
	 * {@link xflaim.DbCheckFlags DbCheckFlags}.
	 * @param checkStatus Optional.  If non-null, then XFlaim will call member
	 * functions to report progress of the check and report any errors that
	 * are found. 
	 * @return Returns an instance of DbInfo containing data on the physical
	 * structure of the database. 
	 * @throws XFlaimException
	 */
	public DbInfo dbCheck(
		String			sDbFileName,
		String			sDataDir,
		String			sRflDir,
		String			sPassword,
		int				iFlags,
		DbCheckStatus	checkStatus) throws XFlaimException
	{
		 long	lRef = _dbCheck( m_this, sDbFileName, sDataDir, sRflDir,
		 						 sPassword, iFlags, checkStatus);
		 return new DbInfo( lRef);
	}
	
	/**
	 * Makes a copy of an existing database.
	 * @param sSrcDbName The name of the control file of the database to be copied.
	 * @param sSrcDataDir The directory where the data files for the source
	 * database are stored. See dbCreate for more information.
	 * @param sSrcRflDir The directory where the RFL files for the source
	 * database are stored.  See dbCreate for more information.
	 * @param sDestDbName The name of the control file that is to be created
	 * for the destination database.
	 * @param sDestDataDir The directory where the data files for the
	 * destination database will be stored.  See dbCreate for more information.
	 * @param sDestRflDir The directory where the RFL files for the
	 * destination database will be stored.  See dbCreate for more information.
	 * @param copyStatus If non-null this is an object that implements the
	 * {@link xflaim.DbCopyStatus DbCopyStatus} interface.  It is a callback
	 * object that is used to report copy progress.
	 * @throws XFlaimException
	 */
	public void dbCopy(
		String			sSrcDbName,
		String			sSrcDataDir,
		String			sSrcRflDir,
		String			sDestDbName,
		String			sDestDataDir,
		String			sDestRflDir,
		DbCopyStatus	copyStatus) throws XFlaimException
	{
		_dbCopy( m_this, sSrcDbName, sSrcDataDir, sSrcRflDir,
				 sDestDbName, sDestDataDir, sDestRflDir, copyStatus);
	}

	/**
	 * Renames a database.
	 * @param sDbName The name of the control file of the database to be renamed.
	 * @param sDataDir The data file directory.  See dbCreate for more
	 * information.
	 * @param sRflDir The roll-forward log file directory.  See dbCreate for more
	 * information.
	 * @param sNewDbName The new control file name for the database.
	 * @param bOverwriteDestOk If true, then if the database specified in
	 * sNewDbName already exists, it will be overwritten.
	 * @param renameStatus If non-null this is an object that implements the
	 * {@link xflaim.DbRenameStatus DbRenameStatus} interface.  It is a callback
	 * object that is used to report rename progress.
	 * @throws XFlaimException
	 */
	public void dbRename(
		String				sDbName,
		String				sDataDir,
		String				sRflDir,
		String				sNewDbName,
		boolean				bOverwriteDestOk,
		DbRenameStatus		renameStatus) throws XFlaimException
	{
		_dbRename( m_this, sDbName, sDataDir, sRflDir, sNewDbName,
				   bOverwriteDestOk, renameStatus);
	}

	/**
	 * Rebuilds a database.
	 * @param sSourceDbPath The name of the control file of the database to be
	 * rebuilt.
	 * @param sSourceDataDir The data file directory.  See dbCreate for more
	 * information.
	 * @param sDestDbPath The name of the control file of the destination
	 * database that is to be built from the source database.
	 * @param sDestDataDir The destination database's data file directory.
	 * See dbCreate for more information.
	 * @param sDestRflDir The destination database's roll-forward log file
	 * directory.  See dbCreate for more information.
	 * @param sDictPath The name of a file containing dictionary definitions that
	 * are to be put into the destination database when it is created.
	 * @param sPassword Password for opening the source database.  This is only needed
	 * if the database key is currently wrapped in a password instead of the
	 * local NICI storage key.
	 * @param createOpts - A {@link xflaim.CREATEOPTS CREATEOPTS} object that
	 * contains several parameters that are used in the creation of the
	 * destination database.  (For advanced users.) 
	 * @param rebuildStatus If non-null this is an object that implements the
	 * {@link xflaim.RebuildStatus RebuildStatus} interface.  It is a callback
	 * object that is used to report rebuild progress.
	 * @throws XFlaimException
	 */
	public void dbRebuild(
		String			sSourceDbPath,
		String			sSourceDataDir,
		String			sDestDbPath,
		String			sDestDataDir,
		String			sDestRflDir,
		String			sDictPath,
		String			sPassword,
		CREATEOPTS		createOpts,
		RebuildStatus	rebuildStatus) throws XFlaimException
	{
		_dbRebuild( m_this, sSourceDbPath, sSourceDataDir, sDestDbPath,
						sDestDataDir, sDestRflDir, sDictPath, sPassword,
						createOpts, rebuildStatus);
	}

	/**
	 * Opens an input stream that reads from a string buffer.
	 * @param sBuffer String buffer that is to be used as an input stream.
	 * @return Returns a {@link xflaim.IStream IStream} object.
	 * @throws XFlaimException
	 */
	public IStream openBufferIStream(
		String	sBuffer) throws XFlaimException
	{
		IStream	jIStream = null;
		long		lRef = 0;

		lRef = _openBufferIStream( m_this, sBuffer);
		
		if (lRef != 0)
		{
			jIStream = new IStream( lRef, this);
		}
		
		return( jIStream);
	}

	/**
	 * Opens a file to be used as an input stream.
	 * @param sPath The pathname of the file to be opened.
	 * @return Returns an instance of IStream.
	 * @throws XFlaimException
	 */
	public IStream openFileIStream(
		String	sPath) throws XFlaimException
	{
		IStream	jIStream = null;
		long		lRef = 0;
		
		lRef = _openFileIStream( m_this, sPath);

		if (lRef != 0)
		{
			jIStream = new IStream( lRef, this);		
		}
		
		return( jIStream);
	}

	/**
	 * Open a multi-file input stream.
	 * @param sDirectory Directory where the input files are located.
	 * @return sBaseName Base name of the input files.  Files that constitute the
	 * input stream are sBaseName, sBaseName.00000001, sBaseName.00000002, etc. - where
	 * the extension is a Hex number.
	 * @return Returns an {@link xflaim.IStream IStream} object.
	 * @throws XFlaimException
	 */	
	public IStream openMultiFileIStream(
		String	sDirectory,
		String	sBaseName) throws XFlaimException
	{
		IStream	jIStream = null;
		long		lRef = 0;

		lRef = _openMultiFileIStream( m_this, sDirectory, sBaseName);
		
		if (lRef != 0)
		{
			jIStream = new IStream( lRef, this);
		}
		
		return( jIStream);
	}
	
	/**
	 * Open a buffered input stream from an existing input stream.
	 * @param istream An {@link xflaim.IStream IStream} object that is to be
	 * the input for the buffered stream.
	 * @return iBufferSize The size (in bytes) of the buffer to use for the
	 * input stream.  Data will be read into the buffer in chunks of this size.
	 * This will help performance by preventing lots of smaller reads from
	 * the original input stream.
	 * @return Returns an {@link xflaim.IStream IStream} object.
	 * @throws XFlaimException
	 */	
	public IStream openBufferedIStream(
		IStream		istream,
		int			iBufferSize) throws XFlaimException
	{
		IStream	jIStream = null;
		long		lRef = 0;

		lRef = _openBufferedIStream( m_this, istream.getThis(), iBufferSize);
		if (lRef != 0)
		{
			jIStream = new IStream( lRef, this);
		}
		
		return( jIStream);
	}
	
	/**
	 * Open an input stream that decompresses data from another input stream.  It
	 * is assumed that data coming out of the other input stream is compressed.
	 * @param istream An {@link xflaim.IStream IStream} object that is to be
	 * the input for this input stream.  It is assumed that the data coming out
	 * of this input stream is compressed.
	 * @return Returns an {@link xflaim.IStream IStream} object.
	 * @throws XFlaimException
	 */	
	public IStream openUncompressingIStream(
		IStream	istream) throws XFlaimException
	{
		IStream	jIStream = null;
		long		lRef = 0;

		lRef = _openUncompressingIStream( m_this, istream.getThis());
		if (lRef != 0)
		{
			jIStream = new IStream( lRef, this);
		}
		
		return( jIStream);
	}
	
	/**
	 * Open an input stream that encodes data from another input stream into
	 * base 64 encoded binary.  Data read from the stream object returned by
	 * this method will be base 64 encoded.
	 * @param istream An {@link xflaim.IStream IStream} object that is to be
	 * the input for this input stream.
	 * @param bInsertLineBreaks Flag indicating whether or not line breaks
	 * should be inserted into the data as it is base 64 encoded.
	 * @return Returns an {@link xflaim.IStream IStream} object.
	 * @throws XFlaimException
	 */	
	public IStream openBase64Encoder(
		IStream				istream,
		boolean				bInsertLineBreaks) throws XFlaimException
	{
		IStream	jIStream = null;
		long		lRef = 0;

		lRef = _openBase64Encoder( m_this, istream.getThis(), bInsertLineBreaks);
		if (lRef != 0)
		{
			jIStream = new IStream( lRef, this);
		}
		
		return( jIStream);
	}
	
	/**
	 * Open an input stream that decodes data from another input stream.  It is
	 * assumed that data read from the original input stream is base 64
	 * encoded.
	 * @param istream An {@link xflaim.IStream IStream} object that is to be
	 * the input for this input stream.  It is assumed that data read from this
	 * input stream will come back base 64 encoded.
	 * @return Returns an {@link xflaim.IStream IStream} object.
	 * @throws XFlaimException
	 */	
	public IStream openBase64Decoder(
		IStream	istream) throws XFlaimException
	{
		IStream	jIStream = null;
		long		lRef = 0;

		lRef = _openBase64Decoder( m_this, istream.getThis());
		if (lRef != 0)
		{
			jIStream = new IStream( lRef, this);
		}
		
		return( jIStream);
	}

	/**
	 * Open an output stream that writes data to a file.
	 * @param sFileName File name to write data to.
	 * @param bTruncateIfExists Flag indicating whether or not the output file
	 * should be truncated if it already exists.  If false, the file will be
	 * appended to.
	 * @return Returns an {@link xflaim.OStream OStream} object.
	 * @throws XFlaimException
	 */	
	public OStream openFileOStream(
		String	sFileName,
		boolean	bTruncateIfExists) throws XFlaimException
	{
		OStream	jOStream = null;
		long		lRef = 0;

		lRef = _openFileOStream( m_this, sFileName, bTruncateIfExists);
		if (lRef != 0)
		{
			jOStream = new OStream( lRef, this);
		}
		
		return( jOStream);
	}
	
	/**
	 * Open a multi-file output stream. Data is written to one or more files.
	 * @param sDirectory Directory where output files are to be created.
	 * @param sBaseName Base name for creating file names.  The first file will
	 * be called sBaseName.  Subsequent files will be named sBaseName.00000001,
	 * sBaseName.00000002, etc.  The extension is a hex number.
	 * @param iMaxFileSize Maximum number of bytes to write to each file in the
	 * multi-file set.
	 * @param bOkToOverwrite Flag indicating whether or not the output files
	 * should be overwritten if they already exist.
	 * @return Returns an {@link xflaim.OStream OStream} object.
	 * @throws XFlaimException
	 */	
	public OStream openMultiFileOStream(
		String				sDirectory,
		String				sBaseName,
		int					iMaxFileSize,
		boolean				bOkToOverwrite) throws XFlaimException
	{
		OStream	jOStream = null;
		long		lRef = 0;

		lRef = _openMultiFileOStream( m_this, sDirectory, sBaseName, iMaxFileSize,
												bOkToOverwrite);
		if (lRef != 0)
		{
			jOStream = new OStream( lRef, this);
		}
		
		return( jOStream);
	}
	
	/**
	 * Delete a multi-file stream from disk.
	 * @param sDirectory Directory where the multiple files that constitute the
	 * stream are located.
	 * @param sBaseName Base name for files in the multi-file stream.  The first file will
	 * be called sBaseName.  Subsequent files will be named sBaseName.00000001,
	 * sBaseName.00000002, etc.  The extension is a hex number.
	 * @throws XFlaimException
	 */	
	public void removeMultiFileStream(
		String				sDirectory,
		String				sBaseName) throws XFlaimException
	{
		_removeMultiFileStream( m_this, sDirectory, sBaseName);
	}
	
	/**
	 * Open a buffered output stream.  A buffer is allocated for writing data to
	 * the original output stream.  Instead of writing small chunks of data to
	 * the original output stream, it is first gathered into the output buffer.
	 * When the output buffer fills, the entire buffer is sent to the original
	 * output stream with a single write.  The idea is that by buffering the
	 * output data, performance can be improved.
	 * @param iBufferSize Size of the buffer to be used for buffering output.
	 * @return Returns an {@link xflaim.OStream OStream} object.
	 * @throws XFlaimException
	 */	
	public OStream openBufferedOStream(
		OStream				ostream,
		int					iBufferSize) throws XFlaimException
	{
		OStream	jOStream = null;
		long		lRef = 0;

		lRef = _openBufferedOStream( m_this, ostream.getThis(), iBufferSize);
		if (lRef != 0)
		{
			jOStream = new OStream( lRef, this);
		}
		
		return( jOStream);
	}
	
	/**
	 * Open a compressing output stream.  Data is compressed before writing it
	 * out to the passed in output stream object.
	 * @return Returns an {@link xflaim.OStream OStream} object.
	 * @throws XFlaimException
	 */	
	public OStream openCompressingOStream(
		OStream ostream) throws XFlaimException
	{
		OStream	jOStream = null;
		long		lRef = 0;

		lRef = _openCompressingOStream( m_this, ostream.getThis());
		if (lRef != 0)
		{
			jOStream = new OStream( lRef, this);
		}
		
		return( jOStream);
	}
	
	/**
	 * Read data from an input stream and write it out to an output stream.  This
	 * is a quick way to copy all data from an input stream to an output stream.
	 * @param istream Input stream data is to be read from.
	 * @param ostream Output stream data is to be written to.
	 * @throws XFlaimException
	 */	
	public void writeToOStream(
		IStream	istream,
		OStream	ostream) throws XFlaimException
	{
		_writeToOStream( m_this, istream.getThis(), ostream.getThis());
	}
	
	/**
	 * Creates and returns a DataVector object to be used when searching
	 * indexes.
	 * @return Returns a {@link xflaim.DataVector DataVector} object.
	 * @throws XFlaimException
	 */	
	public DataVector createJDataVector() throws XFlaimException
	{
		DataVector		jDataVector = null;
		long				lRef = 0;
		
		lRef = _createJDataVector( m_this);
							
		if (lRef != 0)
		{
			jDataVector = new DataVector(lRef);
		}
		
		return jDataVector;
	}

	/**
	 * Updates a parameter in the .ini file.
	 * @param sParamName Name of parameter to be updated.
	 * @param sValue Value that is to be assigned to the parameter.
	 * @throws XFlaimException
	 */	
	public void updateIniFile(
		String		sParamName,
		String		sValue) throws XFlaimException
	{
		_updateIniFile( m_this, sParamName, sValue);
	}
	
	/**
	 * Dup a {@link xflaim.Db Db} object.  This method is a quicker way to open
	 * a database than calling the dbOpen method.  If the application has already
	 * opened a database, it may pass the {@link xflaim.Db Db} object it obtained
	 * into this method to get another {@link xflaim.Db Db} object.
	 * @param dbToDup {@link xflaim.Db Db} object to dup.
	 * @return Returns a new {@link xflaim.Db Db} object that is opened to the
	 * same database as dbToDup.
	 * @throws XFlaimException
	 */	
	public Db dbDup(
		Db			dbToDup) throws XFlaimException
	{
		Db 	jDb = null;
		long 	jDb_ref;
											
		if( (jDb_ref = _dbDup( m_this, dbToDup.getThis())) != 0)
		{
			jDb = new Db( jDb_ref, this);
		}
		
		return( jDb);
	}

	/**
	 * Set a cache limit that is dynamically adjusted.
	 * @param iCacheAdjustPercent Percent of available memory that the cache
	 * limit is to be set to.  A new cache limit is periodically recalculated
	 * based on this percentage.
	 * @param iCacheAdjustMin Minimum value that the cache limit is to be set
	 * to whenever a new cache limit is calculated.
	 * @param iCacheAdjustMax Maximum value that the cache limit is to be set
	 * to whenever a new cache limit is calculated.
	 * @param iCacheAdjustMinToLeave This is an alternative way to specify a
	 * maximum cache limit.  If zero, this parameter is ignored and
	 * iCacheAdjustMax is used.  If non-zero, the maximum cache limit is calculated
	 * to be the amount of available memory minus this number - the idea being to
	 * leave a certain amount of memory for other processes to use.
	 * @throws XFlaimException
	 */	
	public void setDynamicMemoryLimit(
		int	iCacheAdjustPercent,
		int	iCacheAdjustMin,
		int	iCacheAdjustMax,
		int	iCacheAdjustMinToLeave) throws XFlaimException
	{
		_setDynamicMemoryLimit( m_this, iCacheAdjustPercent, iCacheAdjustMin,
							iCacheAdjustMax, iCacheAdjustMinToLeave);
	}

	/**
	 * Set a cache limit that is permanent until the next explicit call to either
	 * setHardMemoryLimit or setDynamicMemoryLimit.
	 * @param iPercent  If non-zero, the new cache limit will be calculated as a
	 * percentage of either the available memory or as a percentage of the
	 * total memory on the system.  iMin and iMax and iMinToLeave are used to
	 * determine a minimum and maximum range for the new cache limit.
	 * @param bPercentOfAvail Only used if iPercent is non-zero.  If true, it
	 * specifies that the percent is to be percent of available memory.  If false,
	 * the percent is the percent of total memory on the system.
	 * @param iMin Only used if iPercent is non-zero.  Specifies the minimum
	 * value that the cache limit is to be allowed to be set to.
	 * @param iMax If iPercent is non-zero, this specifies the maxmimum value
	 * that the cache limit is to be set to.  If iPercent is zero, this specifies
	 * the new cache limit (in bytes).
	 * @param iMinToLeave Only used if iPercent is non-zero.  In that
	 * case, and this value is non-zero, this is an alternative way to specify a
	 * maximum cache limit.  If zero, this parameter is ignored and
	 * iMax is used.  If non-zero, the maximum cache limit is calculated
	 * to be the amount of available memory (or total memory if bPercentOfAvail
	 * is false) minus this number - the idea being to leave a certain amount
	 * of memory for other processes to use.
	 * @param bPreallocate Flag indicating whether cache should be
	 * pre-allocated.  If true, the amount of memory specified in the new
	 * limit will be allocated immediately.  Otherwise, the memory is allocated
	 * as needed.
	 * @throws XFlaimException
	 */	
	public void setHardMemoryLimit(
		int		iPercent,
		boolean	bPercentOfAvail,
		int		iMin,
		int		iMax,
		int		iMinToLeave,
		boolean	bPreallocate) throws XFlaimException
	{
		_setHardMemoryLimit( m_this, iPercent, bPercentOfAvail, iMin, iMax,
					iMinToLeave, bPreallocate);
	}

	/**
	 * Determine if dynamic cache limits are supported on this platform.
	 * @return Boolean indicating whether or not dynamic cache limits are
	 * supported on this platform.
	 * @throws XFlaimException
	 */	
	public boolean getDynamicCacheSupported() throws XFlaimException
	{
		return( _getDynamicCacheSupported( m_this));
	}

	/**
	 * Return information about current cache usage and other cache
	 * statistics.
	 * @return Returns a {@link xflaim.CacheInfo CacheInfo} object.
	 * @throws XFlaimException
	 */	
	public CacheInfo getCacheInfo() throws XFlaimException
	{
		return( _getCacheInfo( m_this));
	}

	/**
	 * Enable or disable debugging of XFLAIM cache.
	 * @param bDebug  If true, enable debugging, otherwise disable.
	 * @throws XFlaimException
	 */	
	public void enableCacheDebug(
		boolean	bDebug) throws XFlaimException
	{
		_enableCacheDebug( m_this, bDebug);
	}

	/**
	 * Determine if cache debugging is currently enabled.
	 * @return Boolean indicating whether or not cache debugging is currently
	 * enabled.
	 * @throws XFlaimException
	 */	
	public boolean cacheDebugEnabled() throws XFlaimException
	{
		return( _cacheDebugEnabled( m_this));
	}

	/**
	 * Close all file descriptors that have are not currently in use and have been
	 * out of use for at least n seconds.
	 * @param iSeconds Specifies the number of seconds.  File descriptors that
	 * are not currently in use and have been out of use for at least this amount
	 * of time will be closed and released.  A value of zero will cause all file
	 * descriptors not currently in use to be closed and released.
	 * @throws XFlaimException
	 */	
	public void closeUnusedFiles(
		int		iSeconds) throws XFlaimException
	{
		_closeUnusedFiles( m_this, iSeconds);
	}

	/**
	 * Start collecting of statistics.
	 * @throws XFlaimException
	 */	
	public void startStats() throws XFlaimException
	{
		_startStats( m_this);
	}

	/**
	 * Stop collecting of statistics.  NOTE: Statistics collected from the time
	 * the startStats method was called will still be available to retrieve
	 * from the getStats() method.
	 * @throws XFlaimException
	 */	
	public void stopStats() throws XFlaimException
	{
		_stopStats( m_this);
	}

	/**
	 * Reset statistics.  All current statistics are started over - as if the
	 * startStats method had been called.
	 * @throws XFlaimException
	 */	
	public void resetStats() throws XFlaimException
	{
		_resetStats( m_this);
	}

	/**
	 * Retrieve the current statistics that have been collected so far.
	 * @return Returns a {@link xflaim.Stats Stats} object.
	 * @throws XFlaimException
	 */	
	public Stats getStats() throws XFlaimException
	{
		return( _getStats( m_this));
	}
	
	/**
	 * Set the directory where temporary files are to be created.
	 * @param sPath Name of directory where temporary files are to be created.
	 * @throws XFlaimException
	 */	
	public void setTempDir(
		String	sPath) throws XFlaimException
	{
		_setTempDir( m_this, sPath);
	}

	/**
	 * Get the directory where temporary files are to be created.
	 * @return Returns the name of directory where temporary files are to be created.
	 * @throws XFlaimException
	 */	
	public String getTempDir() throws XFlaimException
	{
		return( _getTempDir( m_this));
	}

	/**
	 * Set the checkpoint interval.  The checkpoint interval is the maximum number
	 * of seconds that XFLAIM will allow to go by before a checkpoint is forced.
	 * Note that XFLAIM attempt to complete a checkpoint as often as possible.
	 * However, if many update transctions are being performed one after the other
	 * with no break, it is possible that XFLAIM will not be able to complete
	 * a checkpoint.  If the checkpoint interval is exceeded without a checkpoint
	 * being done, XFLAIM will hold off updaters until a checkpoint can be
	 * completed.  This is what is known as a "forced" checkpoint.
	 * @param iSeconds Checkpoint interval, in seconds.
	 * @throws XFlaimException
	 */	
	public void setCheckpointInterval(
		int		iSeconds) throws XFlaimException
	{
		_setCheckpointInterval( m_this, iSeconds);
	}

	/**
	 * Get the current checkpoint interval.
	 * @return Returns the current checkpoint interval, in seconds.
	 * @throws XFlaimException
	 */	
	public int getCheckpointInterval() throws XFlaimException
	{
		return( _getCheckpointInterval( m_this));
	}

	/**
	 * Set the cache adjust interval.  The cache adjust interval is only used
	 * when the application has set a dynamic cache limit (see the
	 * setDynamicCacheLimit API).  It specifies how often XFLAIM should calculate
	 * a new cache limit.
	 * @param iSeconds Specifies the number of seconds between times when XFLAIM
	 * recalculates a new cache limit.
	 * @throws XFlaimException
	 */	
	public void setCacheAdjustInterval(
		int		iSeconds) throws XFlaimException
	{
		_setCacheAdjustInterval( m_this, iSeconds);
	}

	/**
	 * Get the current cache adjust interval.
	 * @return Returns the current cache adjust interval, in seconds.
	 * @throws XFlaimException
	 */	
	public int getCacheAdjustInterval() throws XFlaimException
	{
		return( _getCacheAdjustInterval( m_this));
	}

	/**
	 * Set the current cache cleanup interval.  XFLAIM has a background thread
	 * that periodically wakes up and removes "old" objects from cache.  Old
	 * objects are objects that are prior versions of current objects.  During
	 * a cleanup cycle, XFLAIM determines which of these objects are never going
	 * to be needed again and removes them from cache.
	 * @param iSeconds Specifies the number of seconds between times when XFLAIM
	 * cleans up "old" objects in cache.
	 * @throws XFlaimException
	 */	
	public void setCacheCleanupInterval(
		int		iSeconds) throws XFlaimException
	{
		_setCacheCleanupInterval( m_this, iSeconds);
	}

	/**
	 * Get the current cache cleanup interval.
	 * @return Returns the current cache cleanup interval, in seconds.
	 * @throws XFlaimException
	 */	
	public int getCacheCleanupInterval() throws XFlaimException
	{
		return( _getCacheCleanupInterval( m_this));
	}

	/**
	 * Set the current unused cleanup interval.  XFLAIM has a background thread
	 * that periodically wakes up and removes objects that have not been in use
	 * for a certain amount of time (as specified by the setMaxUnusedTime method).
	 * This includes file descriptors and other in-memory objects that XFLAIM
	 * may have been holding on to in case they are reused.  It does NOT include
	 * blocks in block cache or nodes in node cache.
	 * @param iSeconds Specifies the number of seconds between times when XFLAIM
	 * cleans up "unused" objects in cache.
	 * @throws XFlaimException
	 */	
	public void setUnusedCleanupInterval(
		int		iSeconds) throws XFlaimException
	{
		_setUnusedCleanupInterval( m_this, iSeconds);
	}

	/**
	 * Get the current unused cleanup interval.
	 * @return Returns the current unused cleanup interval, in seconds.
	 * @throws XFlaimException
	 */	
	public int getUnusedCleanupInterval() throws XFlaimException
	{
		return( _getUnusedCleanupInterval( m_this));
	}

	/**
	 * Set the maximum unused time limit.  XFLAIM has a background thread
	 * that periodically wakes up and removes objects that have not been in use
	 * for a certain amount of time.  This includes file descriptors and
	 * other in-memory objects that XFLAIM may have been holding on to in case
	 * they are reused.  This method allows an application to specify a timeout
	 * value that determines the maximum time an object may be "unused" before
	 * it is cleaned up.
	 * @param iSeconds Specifies the time limit (in seconds) for objects to be
	 * "unused" before they are cleaned up.
	 * @throws XFlaimException
	 */	
	public void setMaxUnusedTime(
		int		iSeconds) throws XFlaimException
	{
		_setMaxUnusedTime( m_this, iSeconds);
	}

	/**
	 * Get the maximum unused time limit.
	 * @return Returns the maximum unused time limit, in seconds.
	 * @throws XFlaimException
	 */	
	public int getMaxUnusedTime() throws XFlaimException
	{
		return( _getMaxUnusedTime( m_this));
	}

	/**
	 * Deactivate an open database.  This method allows an application to force
	 * a particular database to be closed by all threads.
	 * @param sDbFileName The name of the control file of the database to.
	 * deactivate.  For more explanation see documentation for dbCreate.
	 * @param sDataDir The data file directory.  See dbCreate for more
	 * information.
	 * @throws XFlaimException
	 */	
	public void deactivateOpenDb(
		String	sDbFileName,
		String	sDataDir) throws XFlaimException
	{
		_deactivateOpenDb( m_this, sDbFileName, sDataDir);
	}
	
	/**
	 * Set maximum number of queries to save statistics and information on.  NOTE:
	 * If the startStats method is called, the maximum is set to 20 until
	 * stopStats is called - unless a non-zero value has already been set.
	 * @param iMaxToSave The maximum number of queries to save information on.  The
	 * last N queries that were executed will be saved.
	 * @throws XFlaimException
	 */	
	public void setQuerySaveMax(
		int		iMaxToSave) throws XFlaimException
	{
		_setQuerySaveMax( m_this, iMaxToSave);
	}

	/**
	 * Get maximum number of queries to save statistics and information on.
	 * @return Returns the maximum number of queries to save information on.
	 * @throws XFlaimException
	 */	
	public int getQuerySaveMax() throws XFlaimException
	{
		return( _getQuerySaveMax( m_this));
	}

	/**
	 * Set dirty cache limits.
	 * @param iMaxDirty This is the maximum amount of cache (in bytes) that the system
	 * should allow to be dirty.  Once the maximum is exceeded, XFLAIM will
	 * attempt to write out dirty blocks until the dirty cache is less than or
	 * equal to the value specified by iLowDirty.
	 * @param iLowDirty This number is the low threshhold for dirty cache.  It is
	 * a hysteresis value.  Once iMaxDirty is exceeded, XFLAIM will write out
	 * dirty blocks until the dirty cache is once again less than or equal to
	 * this number.
	 * @throws XFlaimException
	 */	
	public void setDirtyCacheLimits(
		int		iMaxDirty,
		int		iLowDirty) throws XFlaimException
	{
		_setDirtyCacheLimits( m_this, iMaxDirty, iLowDirty);
	}

	/**
	 * Get the maximum dirty cache limit.  See setDirtyCacheLimits for an
	 * explanation of what the maximum dirty cache limit is.
	 * @return Returns the maximum dirty cache limit.
	 * @throws XFlaimException
	 */	
	public int getMaxDirtyCacheLimit() throws XFlaimException
	{
		return( _getMaxDirtyCacheLimit( m_this));
	}
		
	/**
	 * Get the low dirty cache limit.  See setDirtyCacheLimits for an
	 * explanation of what the low dirty cache limit is.
	 * @return Returns the low dirty cache limit.
	 * @throws XFlaimException
	 */	
	public int getLowDirtyCacheLimit() throws XFlaimException
	{
		return( _getLowDirtyCacheLimit( m_this));
	}

	/**
	 * Compare two strings.
	 * @param sLeftString This is the string on the left side of the comparison
	 * operation.
	 * @param bLeftWild This flag, if true, specifies that wildcard characters
	 * found in sLeftString  should be treated as wildcard characters instead of
	 * literal characters to compare.  If false, the wildcard character (*) is
	 * treated like a normal character.
	 * @param sRightString This is the string on the right side of the comparison
	 * operation.
	 * @param bRightWild This flag, if true, specifies that wildcard characters
	 * found in sRightString should be treated as wildcard characters instead of
	 * literal characters to compare.  If false, the wildcard character (*) is
	 * treated like a normal character.
	 * @param iCompareRules Flags for doing string comparisons.  Should be
	 * logical ORs of the members of {@link xflaim.CompareRules CompareRules}.
	 * @param iLanguage Language to use for doing collation of strings.
	 * @return Returns a value indicating whether sLeftString is less than, equal to,
	 * or greater than sRightString.  A value of -1 means sLeftString < sRightString.
	 * A value of 0 means the strings are equal.  A value of 1 means that
	 * sLeftString > sRightString.
	 * @throws XFlaimException
	 */	
	public int compareStrings(
		String			sLeftString,
		boolean			bLeftWild,
		String			sRightString,
		boolean			bRightWild,
		int				iCompareRules,
		int				iLanguage) throws XFlaimException
	{
		return( _compareStrings( m_this, sLeftString, bLeftWild,
						sRightString, bRightWild, iCompareRules, iLanguage));
	}
	
	/**
	 * Determine if a string has a sub-string in it.
	 * @param sString This is the string that is to be checked to see if it
	 * contains a substring.
	 * @param sSubString  This is the substring that is being looked for.
	 * @param iCompareRules Flags for doing string comparisons.  Should be
	 * logical ORs of the members of {@link xflaim.CompareRules CompareRules}.
	 * @param iLanguage Language to use for doing collation of strings.
	 * @return Returns a boolean value indicating whether sString contains the
	 * substring specified by sSubString.
	 * @throws XFlaimException
	 */	
	public boolean hasSubStr(
		String			sString,
		String			sSubString,
		int				iCompareRules,
		int				iLanguage) throws XFlaimException
	{
		return( _hasSubStr( m_this, sString, sSubString, iCompareRules, iLanguage));
	}
	
	/**
	 * Determine if a character is an upper-case character.
	 * @param uzChar This is the character that is to be tested.
	 * @return Returns a boolean value indicating whether uzChar is upper
	 * case.
	 * @throws XFlaimException
	 */	
	public boolean uniIsUpper(
		char				uzChar) throws XFlaimException
	{
		return( _uniIsUpper( m_this, uzChar));
	}
	
	/**
	 * Determine if a character is a lower-case character.
	 * @param uzChar This is the character that is to be tested.
	 * @return Returns a boolean value indicating whether uzChar is lower
	 * case.
	 * @throws XFlaimException
	 */	
	public boolean _uniIsLower(
		char				uzChar) throws XFlaimException
	{
		return( _uniIsLower( m_this, uzChar));
	}
	
	/**
	 * Determine if a character is an alphabetic character.
	 * @param uzChar This is the character that is to be tested.
	 * @return Returns a boolean value indicating whether uzChar is
	 * alphabetic.
	 * @throws XFlaimException
	 */	
	public boolean uniIsAlpha(
		char				uzChar) throws XFlaimException
	{
		return( _uniIsAlpha( m_this, uzChar));
	}
	
	/**
	 * Determine if a character is a a decimal digit (0..9).
	 * @param uzChar This is the character that is to be tested.
	 * @return Returns a boolean value indicating whether uzChar is
	 * a decimal digit.
	 * @throws XFlaimException
	 */	
	public boolean uniIsDecimalDigit(
		char				uzChar) throws XFlaimException
	{
		return( _uniIsDecimalDigit( m_this, uzChar));
	}
	
	/**
	 * Convert a character to lower case.
	 * @param uzChar This is the character that is to be converted.
	 * @return Returns the lower-case character.
	 * @throws XFlaimException
	 */	
	public char uniToLower(
		char				uzChar) throws XFlaimException
	{
		return( _uniToLower( m_this, uzChar));
	}
	
	/**
	 * Wait for a database to close.  This method will not return until the
	 * database specified has been closed by all Db ojects that currently have
	 * it open.
	 * @param sDbFileName The name of the control file of the database to wait
	 * to close.  For more explanation see documentation for dbCreate.
	 * @throws XFlaimException
	 */	
	public void waitToClose(
		String			sDbFileName) throws XFlaimException
	{
		_waitToClose( m_this, sDbFileName);
	}

	/**
	 * Free as much cache as possible. NOTE: This method will not be able to
	 * remove cached blocks and nodes that are currently in use.
	 * @param dbWithUpdateTrans This is a {@link xflaim.Db Db} object that may be used to
	 * write out dirty cache blocks.  It may be null.  If non-null, it must be
	 * the {@link xflaim.Db Db} object that is currently running an update
	 * transaction.
	 * @throws XFlaimException
	 */	
	public void clearCache(
		Db					dbWithUpdateTrans) throws XFlaimException
	{
		if (dbWithUpdateTrans != null)
		{
			_clearCache( m_this, dbWithUpdateTrans.getThis());
		}
		else
		{
			_clearCache( m_this, 0);
		}
	}
	
// PRIVATE METHODS

	private native long _createDbSystem();
	
	private native void _release( long lThis);

	private native long _dbCreate(
		long			lThis,
		String 		sDbFileName,
		String 		sDataDir,
		String 		sRflDir,
		String 		sDictFileName,
		String 		sDictBuf,
		CREATEOPTS  createOpts);

	private native long _dbOpen(
		long			lThis,
		String		sDbFileName,
		String		sDataDir,
		String		sRflDir,
		String		sPassword,
		boolean		bAllowLimited);

	private native void _dbRemove(
		long			lThis,
		String		sDbFileName,
		String		sDataDir,
		String		sRflDir,
		boolean		bRemoveRflFiles) throws XFlaimException;

	private native void _dbRestore(
		long					lThis,
		String				sDbPath,
		String				sDataDir,
		String				sRflDir,
		String				sBackupPath,
		String				sPassword,
		RestoreClient		RestoreClient,
		RestoreStatus		RestoreStatus) throws XFlaimException;
		
	private native long _dbCheck(
		long					lThis,
		String				sDbFileName,
		String				sDataDir,
		String				sRflDir,
		String				sPassword,
		int					iFlags,
		DbCheckStatus		checkStatus) throws XFlaimException;

	private native void _dbCopy(
		long					lThis,
		String				sSrcDbName,
		String				sSrcDataDir,
		String				sSrcRflDir,
		String				sDestDbName,
		String				sDestDataDir,
		String				sDestRflDir,
		DbCopyStatus		copyStatus) throws XFlaimException;

	private native void _dbRename(
		long					lThis,
		String				sDbName,
		String				sDataDir,
		String				sRflDir,
		String				sNewDbName,
		boolean				bOverwriteDestOk,
		DbRenameStatus		renameStatus) throws XFlaimException;
		
	private native void _dbRebuild(
		long						lThis,
		String					sSourceDbPath,
		String					sSourceDataDir,
		String					sDestDbPath,
		String					sDestDataDir,
		String					sDestRflDir,
		String					sDictPath,
		String					sPassword,
		CREATEOPTS				createOpts,
		RebuildStatus			rebuildStatus) throws XFlaimException;

	private native long _openBufferIStream(
		long					lThis,
		String				sBuffer) throws XFlaimException;

	private native long _openFileIStream(
		long					lThis,
		String				sPath);

	private native long _openMultiFileIStream(
		long			lThis,
		String		sDirectory,
		String		sBaseName) throws XFlaimException;
	
	private native long _openBufferedIStream(
		long					lThis,
		long					lIStream,
		int					iBufferSize) throws XFlaimException;

	private native long _openUncompressingIStream(
		long					lThis,
		long					lIStream) throws XFlaimException;
	
	private native long _openBase64Encoder(
		long					lThis,
		long					lIstream,
		boolean				bInsertLineBreaks) throws XFlaimException;

	private native long _openBase64Decoder(
		long					lThis,
		long					lIstream) throws XFlaimException;

	private native long _openFileOStream(
		long					lThis,
		String				sFileName,
		boolean				bTruncateIfExists) throws XFlaimException;

	private native long _openMultiFileOStream(
		long					lThis,
		String				sDirectory,
		String				sBaseName,
		int					iMaxFileSize,
		boolean				bOkToOverwrite) throws XFlaimException;
	
	private native void _removeMultiFileStream(
		long					lThis,
		String				sDirectory,
		String				sBaseName) throws XFlaimException;
	
	private native long _openBufferedOStream(
		long					lThis,
		long					lOStream,
		int					iBufferSize) throws XFlaimException;
	
	private native long _openCompressingOStream(
		long					lThis,
		long					lOStream) throws XFlaimException;
	
	private native void _writeToOStream(
		long					lThis,
		long					lIstream,
		long					lOStream) throws XFlaimException;
	
	private native long _createJDataVector(
		long					lRef);

	private native void _updateIniFile(
		long			lThis,
		String		sParamName,
		String		sValue) throws XFlaimException;

	private native long _dbDup(
		long			lThis,
		long			lDbToDup) throws XFlaimException;

	private native void _setDynamicMemoryLimit(
		long	lThis,
		int	iCacheAdjustPercent,
		int	iCacheAdjustMin,
		int	iCacheAdjustMax,
		int	iCacheAdjustMinToLeave) throws XFlaimException;

	private native void _setHardMemoryLimit(
		long		lThis,
		int		iPercent,
		boolean	bPercentOfAvail,
		int		iMin,
		int		iMax,
		int		iMinToLeave,
		boolean	bPreallocate) throws XFlaimException;

	private native boolean _getDynamicCacheSupported(
		long		lThis) throws XFlaimException;

	private native CacheInfo _getCacheInfo(
		long		lThis) throws XFlaimException;

	private native void _enableCacheDebug(
		long		lThis,
		boolean	bDebug) throws XFlaimException;

	private native boolean _cacheDebugEnabled(
		long		lThis) throws XFlaimException;

	private native void _closeUnusedFiles(
		long		lThis,
		int		iSeconds) throws XFlaimException;

	private native void _startStats(
		long		lThis) throws XFlaimException;

	private native void _stopStats(
		long		lThis) throws XFlaimException;

	private native void _resetStats(
		long		lThis) throws XFlaimException;

	private native Stats _getStats(
		long		lThis) throws XFlaimException;

	private native void _setTempDir(
		long		lThis,
		String	sPath) throws XFlaimException;

	private native String _getTempDir(
		long		lThis) throws XFlaimException;

	private native void _setCheckpointInterval(
		long		lThis,
		int		iSeconds) throws XFlaimException;

	private native int _getCheckpointInterval(
		long		lThis) throws XFlaimException;

	private native void _setCacheAdjustInterval(
		long		lThis,
		int		iSeconds) throws XFlaimException;

	private native int _getCacheAdjustInterval(
		long		lThis) throws XFlaimException;

	private native void _setCacheCleanupInterval(
		long		lThis,
		int		iSeconds) throws XFlaimException;

	private native int _getCacheCleanupInterval(
		long		lThis) throws XFlaimException;

	private native void _setUnusedCleanupInterval(
		long		lThis,
		int		iSeconds) throws XFlaimException;

	private native int _getUnusedCleanupInterval(
		long		lThis) throws XFlaimException;

	private native void _setMaxUnusedTime(
		long		lThis,
		int		iSeconds) throws XFlaimException;

	private native int _getMaxUnusedTime(
		long		lThis) throws XFlaimException;

	private native void _deactivateOpenDb(
		long		lThis,
		String	sDbFileName,
		String	sDataDir) throws XFlaimException;

	private native void _setQuerySaveMax(
		long		lThis,
		int		iMaxToSave) throws XFlaimException;

	private native int _getQuerySaveMax(
		long		lThis) throws XFlaimException;

	private native void _setDirtyCacheLimits(
		long		lThis,
		int		iMaxDirty,
		int		iLowDirty) throws XFlaimException;

	private native int _getMaxDirtyCacheLimit(
		long		lThis) throws XFlaimException;
		
	private native int _getLowDirtyCacheLimit(
		long		lThis) throws XFlaimException;

	private native int _compareStrings(
		long				lThis,
		String			sLeftString,
		boolean			bLeftWild,
		String			sRightString,
		boolean			bRightWild,
		int				iCompareRules,
		int				iLanguage) throws XFlaimException;
	
	private native boolean _hasSubStr(
		long				lThis,
		String			sString,
		String			sSubString,
		int				iCompareRules,
		int				iLanguage) throws XFlaimException;

	private native boolean _uniIsUpper(
		long				lThis,
		char				uzChar) throws XFlaimException;
	
	private native boolean _uniIsLower(
		long				lThis,
		char				uzChar) throws XFlaimException;
	
	private native boolean _uniIsAlpha(
		long				lThis,
		char				uzChar) throws XFlaimException;
	
	private native boolean _uniIsDecimalDigit(
		long				lThis,
		char				uzChar) throws XFlaimException;
	
	private native char _uniToLower(
		long				lThis,
		char				uzChar) throws XFlaimException;
	
	private native void _waitToClose(
		long				lThis,
		String			sDbFileName) throws XFlaimException;

	private native void _clearCache(
		long				lThis,
		long				lDbRef) throws XFlaimException;

	private long			m_this;
}

/*

METHODS NOT YET IMPLEMENTED

virtual const char * FLMAPI checkErrorToStr(
	FLMINT	iCheckErrorCode) = 0;

virtual void FLMAPI setLogger(
	IF_LoggerClient *		pLogger) = 0;

virtual void FLMAPI enableExtendedServerMemory(
	FLMBOOL					bEnable) = 0;

virtual FLMBOOL FLMAPI extendedServerMemoryEnabled( void) = 0;

virtual RCODE FLMAPI registerForEvent(
	eEventCategory			eCategory,
	IF_EventClient *		ifpEventClient) = 0;

virtual void FLMAPI deregisterForEvent(
	eEventCategory			eCategory,
	IF_EventClient *		ifpEventClient) = 0;

virtual RCODE FLMAPI getNextMetaphone(
	IF_IStream *			ifpIStream,
	FLMUINT *				puiMetaphone,
	FLMUINT *				puiAltMetaphone = NULL) = 0;
	
*/

