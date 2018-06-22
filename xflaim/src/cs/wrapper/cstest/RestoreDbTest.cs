//------------------------------------------------------------------------------
// Desc:	Restore database test
// Tabs:	3
//
// Copyright (c) 2006 Novell, Inc. All Rights Reserved.
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
	// Restore database test.
	//--------------------------------------------------------------------------
	public class RestoreDbTest : Tester
	{
		private class MyRestoreStatus : RestoreStatus
		{
			private ulong	m_ulNumTransCommitted;
			private ulong	m_ulNumTransAborted;
			private bool	m_bOutputLines;
		
			public MyRestoreStatus()
			{
				m_ulNumTransCommitted = 0;
				m_ulNumTransAborted = 0;
				System.Console.WriteLine( " ");
				m_bOutputLines = false;
			}

			public bool outputLines()
			{
				return( m_bOutputLines);
			}

			public RCODE reportProgress(
				ref RestoreAction 	peRestoreAction,
				ulong						ulBytesToDo,
				ulong						ulBytesDone)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;

				System.Console.Write( "Bytes To Restore: {0}, Bytes Restored: {1}, TRCmit: {2}, TRAbrt: {3}\r",
					ulBytesToDo, ulBytesDone, m_ulNumTransCommitted, m_ulNumTransAborted);
				m_bOutputLines = true;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportError(
				ref RestoreAction 	peRestoreAction,
				RCODE						rcErr)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;

				System.Console.WriteLine( "\nError reported: {0}", rcErr);
				m_bOutputLines = true;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportBeginTrans(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;

				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportCommitTrans(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				m_ulNumTransCommitted++;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportAbortTrans(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				m_ulNumTransAborted++;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportBlockChainFree(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				ulong						ulMaintDocNum,
				uint						uiStartBlkAddr,
				uint						uiEndBlkAddr,
				uint						uiCount)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportIndexSuspend(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiIndexNum)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportIndexResume(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiIndexNum)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportReduce(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCount)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportUpgrade(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiOldDbVersion,
				uint						uiNewDbVersion)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportOpenRflFile(
				ref RestoreAction 	peRestoreAction,
				uint						uiFileNum)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportRflRead(
				ref RestoreAction 	peRestoreAction,
				uint						uiFileNum,
				uint						uiBytesRead)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportEnableEncryption(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportWrapKey(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportSetNextNodeId(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulNextNodeId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportNodeSetMetaValue(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulNodeId,
				ulong						ulMetaValue)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportNodeSetPrefixId(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulNodeId,
				uint						uiAttrNameId,
				uint						uiPrefixId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportNodeFlagsUpdate(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulNodeId,
				uint						uiFlags,
				bool						bAdd)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportAttributeSetValue(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulElementNodeId,
				uint						uiAttrNameId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportNodeSetValue(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulNodeId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportNodeUpdate(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulNodeId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportInsertBefore(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulParentId,
				ulong						ulNewChildId,
				ulong						ulRefChildId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportNodeCreate(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulRefNodeId,
				eDomNodeType			eNodeType,
				uint						uiNameId,
				eNodeInsertLoc			eLocation)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportNodeChildrenDelete(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulNodeId,
				uint						uiNameId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportAttributeDelete(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulElementNodeId,
				uint						uiAttrNameId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportNodeDelete(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulNodeId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportDocumentDone(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId,
				uint						uiCollection,
				ulong						ulDocumentId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}

			public RCODE reportRollOverDbKey(
				ref RestoreAction 	peRestoreAction,
				ulong						ulTransId)
			{
				peRestoreAction = RestoreAction.XFLM_RESTORE_ACTION_CONTINUE;
				return( RCODE.NE_XFLM_OK);
			}
		}

		public bool restoreDbTest(
			string	sDbName,
			string	sBackupPath,
			DbSystem	dbSystem)
		{
			MyRestoreStatus	restoreStatus = null;

			// Try restoring the database

			beginTest( "Restore Database Test (from directory \"" + sBackupPath + "\" to " + sDbName + ")");

			restoreStatus = new MyRestoreStatus();
			try
			{
				dbSystem.dbRestore( sDbName, null, null, sBackupPath, null,
					null, restoreStatus);
			}
			catch (XFlaimException ex)
			{
				endTest( restoreStatus.outputLines(), ex, "restoring database");
				return( false);
			}

			endTest( restoreStatus.outputLines(), true);
			return( true);
		}
	}
}
