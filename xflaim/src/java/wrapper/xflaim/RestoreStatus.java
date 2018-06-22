//------------------------------------------------------------------------------
// Desc:	Restore Status
// Tabs:	3
//
// Copyright (c) 2004-2007 Novell, Inc. All Rights Reserved.
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
 * This interface allows XFlaim's backup subsystem to periodicly pass
 * information about the status of a restore operation (bytes completed and
 * bytes remaining) while the operation is running.  The implementor may do
 * anything it wants with the information, such as using it to update a
 * progress bar or simply ignoring it.
 */
public interface RestoreStatus
{
	/**
	 * 
	 * @param lBytesToDo
	 * @param lBytesDone
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportProgress(
		long		lBytesToDo,
		long		lBytesDone);

	/**
	 * 
	 * @param eErrCode
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportError(
		int		eErrCode);

	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportBeginTrans(
		long		lTransId);

	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportCommitTrans(
		long		lTransId);

	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportAbortTrans(
		long		lTransId);
	
	/**
	 * 
	 * @param lTransId
	 * @param iMaintDocNum
	 * @param iStartBlkAddr
	 * @param iEndBlkAddr
	 * @param iCount
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportBlockChainFree(
		long		lTransId,
		int		iMaintDocNum,
		int		iStartBlkAddr,
		int		iEndBlkAddr,
		int		iCount);

	/**
	 * 
	 * @param lTransId
	 * @param iIndexNum
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportIndexSuspend(
		long		lTransId,
		int		iIndexNum);

	/**
	 * 
	 * @param lTransId
	 * @param iIndexNum
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportIndexResume(
		long		lTransId,
		int		iIndexNum);

	/**
	 * 
	 * @param lTransId
	 * @param iCount
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportReduce(
		long		lTransId,
		int		iCount);

	/**
	 * 
	 * @param lTransId
	 * @param iOldDbVersion
	 * @param iNewDbVersion
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportUpgrade(
		long		lTransId,
		int		iOldDbVersion,
		int		iNewDbVersion);

	/**
	 * 
	 * @param iFileNum
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportOpenRflFile(
		int		iFileNum);

	/**
	 * 
	 * @param iFileNum
	 * @param iBytesRead
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportRflRead(
		int		iFileNum,
		int		iBytesRead);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportEnableEncryption(
		long		lTransId);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportWrapKey(
		long		lTransId);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportSetNextNodeId(
		long		lTransId,
		int		iCollection,
		long		lNextNodeId);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportNodeSetMetaValue(
		long		lTransId,
		int		iCollection,
		long		lNodeId,
		long		lMetaValue);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportNodeSetPrefixId(
		long		lTransId,
		int		iCollection,
		long		lNodeId,
		int		iAttrNameId,
		int		iPrefixId);

	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportNodeFlagsUpdate(
		long		lTransId,
		int		iCollection,
		long		lNodeId,
		int		iFlags,
		boolean	bAdd);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportAttributeSetValue(
		long		lTransId,
		int		iCollection,
		long		lElementNodeId,
		int		iAttrNameId);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportNodeSetValue(
		long		lTransId,
		int		iCollection,
		long		lNodeId);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportNodeUpdate(
		long		lTransId,
		int		iCollection,
		long		lNodeId);

	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportInsertBefore(
		long		lTransId,
		int		iCollection,
		long		lParentId,
		long		lNewChildId,
		long		lRefChildId);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportNodeCreate(
		long		lTransId,
		int		iCollection,
		long		lRefNodeId,
		int		eNodeType,
		int		iNameId,
		int		eLocation);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportNodeChildrenDelete(
		long		lTransId,
		int		iCollection,
		long		lNodeId,
		int		iNameId);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportAttributeDelete(
		long		lTransId,
		int		iCollection,
		long		lElementId,
		int		iAttrNameId);

	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportNodeDelete(
		long		lTransId,
		int		iCollection,
		long		lNodeId);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportDocumentDone(
		long		lTransId,
		int		iCollection,
		long		lNodeId);
		
	/**
	 * 
	 * @param lTransId
	 * @return Returns a status code.  The integer should one of the constants
	 * found in {@link xflaim.RCODE xflaim.RCODE}.
	 * Note that returning anything other than NE_XFLM_OK will cause the
	 * restore operation to abort and an XFLaimException to be thrown.
	 */
	int reportRollOverDbKey(
		long		lTransId);
}
