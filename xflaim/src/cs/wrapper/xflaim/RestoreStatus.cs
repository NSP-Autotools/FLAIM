//------------------------------------------------------------------------------
// Desc:	Restore Status
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
	/// Actions that are returned from <see cref="RestoreStatus"/> methods.
	/// These actions tell the <see cref="DbSystem.dbRestore"/> method
	/// what action to take with respect to the operation that it is
	/// reporting it is about to do.
	/// </summary>
	public enum RestoreAction : int
	{
		/// <summary>Continue restore</summary>
		XFLM_RESTORE_ACTION_CONTINUE = 0,
		/// <summary>Stop restore</summary>
		XFLM_RESTORE_ACTION_STOP,
		/// <summary>Skip operation (future)</summary>
		XFLM_RESTORE_ACTION_SKIP,
		/// <summary>Retry the operation</summary>
		XFLM_RESTORE_ACTION_RETRY
	}
	
	/// <summary>
	/// This interface allows XFlaim's restore subsystem to periodicly pass
	/// information about the status of a restore operation (bytes completed and
	/// bytes remaining) while the operation is running.  The implementor may do
	/// anything it wants with the information, such as using it to update a
	/// progress bar or simply ignoring it.
	/// </summary>
	public interface RestoreStatus
	{

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// progress of the restore operation.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulBytesToDo">Total bytes that need to be restored.</param>
		/// <param name="ulBytesDone">Bytes restored so far.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportProgress(
			ref RestoreAction 	peRestoreAction,
			ulong						ulBytesToDo,
			ulong						ulBytesDone);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// an error condition.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="rcErr">Error code.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportError(
			ref RestoreAction 	peRestoreAction,
			RCODE						rcErr);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a transaction begin packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for transaction that is starting.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportBeginTrans(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a transaction commit packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for committed transaction.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportCommitTrans(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a transaction abort packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for aborted transaction.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportAbortTrans(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId);
		
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a block chain free packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="ulMaintDocNum">
		/// This is the document ID of the document in the database where the
		/// block chain free information is maintained.
		/// </param>
		/// <param name="uiStartBlkAddr">
		/// Start block address of block chain to free.
		/// </param>
		/// <param name="uiEndBlkAddr">
		/// Last block in block chain that should be freed.
		/// </param>
		/// <param name="uiCount">
		/// Count of blocks to free.
		/// </param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportBlockChainFree(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			ulong						ulMaintDocNum,
			uint						uiStartBlkAddr,
			uint						uiEndBlkAddr,
			uint						uiCount);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that an index suspend packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiIndexNum">Index being suspended.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportIndexSuspend(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiIndexNum);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that an index resume packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiIndexNum">Index being resumed.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportIndexResume(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiIndexNum);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that an reduce database size packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCount">Count of blocks to reduce.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportReduce(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCount);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that an upgrade database packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiOldDbVersion">Old version database is being upgraded from.</param>
		/// <param name="uiNewDbVersion">New version database is being upgraded to.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportUpgrade(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiOldDbVersion,
			uint						uiNewDbVersion);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a roll-forward log file is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="uiFileNum">Roll-forward log file number.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportOpenRflFile(
			ref RestoreAction 	peRestoreAction,
			uint						uiFileNum);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a roll-forward log file is being read from.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="uiFileNum">Roll-forward log file number.</param>
		/// <param name="uiBytesRead">Number of bytes that were read.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportRflRead(
			ref RestoreAction 	peRestoreAction,
			uint						uiFileNum,
			uint						uiBytesRead);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that an enable encryption packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportEnableEncryption(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a wrap key packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportWrapKey(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a set next node ID packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection whose next node ID is being set.</param>
		/// <param name="ulNextNodeId">Next node ID that is to be set.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportSetNextNodeId(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNextNodeId);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a set node meta value packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the node belongs to.</param>
		/// <param name="ulNodeId">Node ID whose meta value is being set.</param>
		/// <param name="ulMetaValue">Meta value to be set on the node.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportNodeSetMetaValue(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId,
			ulong						ulMetaValue);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a set node prefix ID packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the node belongs to.</param>
		/// <param name="ulNodeId">Node ID whose prefix ID is being set.</param>
		/// <param name="uiAttrNameId">Attribute of node whose prefix ID is to be set (zero means no attribute).</param>
		/// <param name="uiPrefixId">Prefix ID that is to be set on the specified node.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportNodeSetPrefixId(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId,
			uint						uiAttrNameId,
			uint						uiPrefixId);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a set node flags packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the node belongs to.</param>
		/// <param name="ulNodeId">Node ID whose flags are being set.</param>
		/// <param name="uiFlags">Flags to be set on the node.</param>
		/// <param name="bAdd">Specifies whether flags are being set (true) or unset (false).</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportNodeFlagsUpdate(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId,
			uint						uiFlags,
			bool						bAdd);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a set attribute value packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the node belongs to.</param>
		/// <param name="ulElementNodeId">Node ID of element the attribute belongs to whose value is being set.</param>
		/// <param name="uiAttrNameId">Attribute ID whose value is being set.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportAttributeSetValue(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulElementNodeId,
			uint						uiAttrNameId);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a set node value packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the node belongs to.</param>
		/// <param name="ulNodeId">Node ID whose value is being set.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportNodeSetValue(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a node update packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the node belongs to.</param>
		/// <param name="ulNodeId">Node ID of node being updated.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportNodeUpdate(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a node insert packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the node belongs to.</param>
		/// <param name="ulParentId">ID of parent node the new node is being inserted as a child to.</param>
		/// <param name="ulNewChildId">ID of new child node being inserted.</param>
		/// <param name="ulRefChildId">ID of existing child node that the new child node is being inserted as a sibling to.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportInsertBefore(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulParentId,
			ulong						ulNewChildId,
			ulong						ulRefChildId);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a node create packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the node belongs to.</param>
		/// <param name="ulRefNodeId">ID of node the new node will be created relative to.</param>
		/// <param name="eNodeType">Type of node being created.</param>
		/// <param name="uiNameId">Name ID to be given to the new node.</param>
		/// <param name="eLocation">Relative location the new node is to be inserted with respect to the reference node.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportNodeCreate(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulRefNodeId,
			eDomNodeType			eNodeType,
			uint						uiNameId,
			eNodeInsertLoc			eLocation);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a node children delete packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the node belongs to.</param>
		/// <param name="ulNodeId">Node ID of node whose children are to be deleted.</param>
		/// <param name="uiNameId">Name ID of child nodes to be deleted.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportNodeChildrenDelete(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId,
			uint						uiNameId);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that an attribute delete packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the node belongs to.</param>
		/// <param name="ulElementNodeId">Node ID of element whose attribute is being deleted.</param>
		/// <param name="uiAttrNameId">Name ID of attribute being deleted.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportAttributeDelete(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulElementNodeId,
			uint						uiAttrNameId);

		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a node delete packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the node belongs to.</param>
		/// <param name="ulNodeId">Node ID of node being deleted.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportNodeDelete(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulNodeId);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a document done packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <param name="uiCollection">Collection the document belongs to.</param>
		/// <param name="ulDocumentId">Document ID of document that is done.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportDocumentDone(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId,
			uint						uiCollection,
			ulong						ulDocumentId);
			
		/// <summary>
		/// The <see cref="DbSystem.dbRestore"/> method calls this method to report
		/// a that a roll over database key packet is being restored.
		/// </summary>
		/// <param name="peRestoreAction">
		/// Action to be taken by <see cref="DbSystem.dbRestore"/> is returned here.
		/// </param>
		/// <param name="ulTransId">Transaction ID for this packet.</param>
		/// <returns>
		/// If the implementation returns anything except RCODE.NE_XFLM_OK
		/// the <see cref="DbSystem.dbRestore"/> operation will abort and throw
		/// an <see cref="XFlaimException"/>.
		/// </returns>
		RCODE reportRollOverDbKey(
			ref RestoreAction 	peRestoreAction,
			ulong						ulTransId);
	}
}
