//------------------------------------------------------------------------------
// Desc:	Database upgrade routines.
// Tabs:	3
//
//		Copyright (c) 1999-2007 Novell, Inc. All Rights Reserved.
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

#include "flaimsys.h"

/****************************************************************************
Desc : Upgrades a database to the latest FLAIM version.
****************************************************************************/
RCODE XFLAPI F_Db::upgrade(
	IF_UpgradeClient *	// pUpgradeClient
	)
{
	RCODE				rc = NE_XFLM_OK;
	FLMBOOL			bStartedTrans = FALSE;
	FLMBOOL			bLockedDatabase = FALSE;
	FLMUINT			uiOldVersion = 0;
	F_Rfl *			pRfl = m_pDatabase->m_pRfl;
	XFLM_DB_HDR *	pUncommittedDbHdr = &m_pDatabase->m_uncommittedDbHdr;
	FLMUINT64		ui64SaveTransId;
	FLMUINT			uiRflToken = 0;
	FLMBOOL			bUpgradeNeeded = FALSE;

	// Lock the database if not already locked.
	// Cannot lose exclusive access between the checkpoint and
	// the update transaction that does the conversion.

	if (!(m_uiFlags & FDB_HAS_FILE_LOCK))
	{
		if (RC_BAD( rc = dbLock( FLM_LOCK_EXCLUSIVE, 0, 15)))
		{
			goto Exit;
		}
		bLockedDatabase = TRUE;
	}

	// Cannot have any transaction already going.

	if (m_eTransType != XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}

	// NOTE: Don't get the current version number until AFTER obtaining
	// the exclusive lock - to make sure nobody else can or will do
	// an upgrade while we are in here.

	uiOldVersion = (FLMUINT)m_pDatabase->m_lastCommittedDbHdr.ui32DbVersion;

	// Verify that we can do the upgrade
	
	switch (uiOldVersion)
	{
		case XFLM_CURRENT_VERSION_NUM:
			break;
		
		default:
			rc = RC_SET( NE_XFLM_UNALLOWED_UPGRADE);
			goto Exit;
	}
	
	if( !bUpgradeNeeded)
	{
		goto Exit;
	}
	
	// Change state of logging OFF to TRUE - don't want anything
	// logged during conversion except for the upgrade packet.

	pRfl->disableLogging( &uiRflToken);
	m_uiFlags |= FDB_UPGRADING;

	ui64SaveTransId = m_pDatabase->m_lastCommittedDbHdr.ui64CurrTransID;

	// Flush everything do disk so that the roll forward log is empty.
	// The upgrade doesn't put any special data in the roll forward log
	// so if the roll forward log had stuff in it, it would roll forward
	// on data that was a newer version - and never work!
	// Start an update transaction and commit it, forcing it to be
	// checkpointed.

	if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	bStartedTrans = TRUE;

	// Don't want this transaction to change the transaction ID because
	// we are only trying to force a checkpoint.  We don't want to change
	// the transaction ID until we have actually done the convert.

	m_pDatabase->m_uncommittedDbHdr.ui64CurrTransID = ui64SaveTransId;
	m_ui64CurrTransID = ui64SaveTransId;

	// Set up things in the FDB to indicate where we should move the
	// checkpoint file number and offset to.  If we are in the middle
	// of a recovery or restore operation, move the pointer forward
	// to just BEFORE the upgrade packet.  Down below when we do the
	// checkpoint at the end of the upgrade, we will move the pointer
	// forward to just AFTER the upgrade packet.

	if (m_uiFlags & FDB_REPLAYING_RFL)
	{
		m_uiUpgradeCPFileNum = pRfl->getCurrFileNum();
		m_uiUpgradeCPOffset = pRfl->getCurrPacketAddress();
	}

	// Commit the transaction, forcing it to be checkpointed.

	bStartedTrans = FALSE;
	if (RC_BAD( rc = commitTrans( 0, TRUE)))
	{
		goto Exit;
	}

	// Start an update transaction for the conversion.

	if (RC_BAD( rc = beginTrans( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	bStartedTrans = TRUE;

	// Make sure that commit does something.

	m_bHadUpdOper = TRUE;

	// NOTE: By this point, all conversions should be complete, except for
	// committing and changing the version number.

	// Log the upgrade packet to the RFL

	pRfl->enableLogging( &uiRflToken);

	// Log the upgrade packet.

	if( RC_BAD( rc = pRfl->logUpgrade( this, uiOldVersion)))
	{
		goto Exit;
	}

	// Turn logging off again

	pRfl->disableLogging( &uiRflToken);

	// Change the FLAIM version number to the new version number.

	pUncommittedDbHdr->ui32DbVersion = XFLM_CURRENT_VERSION_NUM;

	// Commit and force a checkpoint by passing TRUE.
	// Set up things in the FDB to indicate where we should move the
	// checkpoint file number and offset to.  If we are in the middle
	// of a recovery or restore operation, move the pointer forward
	// to just AFTER the upgrade packet.

	if (m_uiFlags & FDB_REPLAYING_RFL)
	{
		m_uiUpgradeCPFileNum = pRfl->getCurrFileNum();
		m_uiUpgradeCPOffset = pRfl->getCurrReadOffset();
	}

	bStartedTrans = FALSE;
	if (RC_BAD( rc = commitTrans( 0, TRUE)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		// Failure condition, we jumped to exit

		pUncommittedDbHdr->ui32DbVersion = (FLMUINT32)uiOldVersion;
		m_pDatabase->m_lastCommittedDbHdr.ui32DbVersion =
					(FLMUINT32)uiOldVersion;

		(void)abortTrans();
	}

	if (uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	// Turn off the upgrade flag, in case it was turned on above.

	m_uiFlags &= (~(FDB_UPGRADING));

	if (bLockedDatabase)
	{
		(void)dbUnlock();
	}

	return( rc );
}

/************************************************************************
Desc : Enable encryption on the database.
*************************************************************************/
RCODE XFLAPI F_Db::enableEncryption( void)
{
	RCODE					rc = NE_XFLM_OK;
	F_Rfl *				pRfl = m_pDatabase->m_pRfl;
	FLMBYTE *			pucWrappingKey = NULL;
	FLMUINT32			ui32KeyLen = 0;
	XFLM_DB_HDR *		pucUncommittedLogHdr = &m_pDatabase->m_uncommittedDbHdr;
	FLMBOOL				bLocked = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMUINT				uiRflToken = 0;

	// We must will start our own transaction

	if( m_eTransType != XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}

	if (!(m_uiFlags & FDB_HAS_FILE_LOCK))
	{
		if ( RC_BAD( rc = dbLock( FLM_LOCK_EXCLUSIVE, 0, FLM_NO_TIMEOUT)))
		{
			goto Exit;
		}

		bLocked = TRUE;
	}

	// Disable RFL logging

	pRfl->disableLogging( &uiRflToken);

	// Begin an update transaction.

	if (RC_BAD( rc = transBegin( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}

	bStartedTrans = TRUE;

	// If we don't have a wrapping key, then create one.  Normally
	// this would be the case, since we are enabling encryption,
	// but the test is "just to be sure" we don't
	// overwrite an existing key.

	if (!m_pDatabase->m_pWrappingKey)
	{
		if ( RC_BAD( rc = createDbKey()))
		{
			goto Exit;
		}
	}

	if( RC_BAD( rc = m_pDatabase->m_pWrappingKey->getKeyToStore(
		&pucWrappingKey, &ui32KeyLen, 
		m_pDatabase->m_pszDbPasswd, NULL)))
	{
		goto Exit;
	}
	
	f_memcpy( pucUncommittedLogHdr->DbKey, pucWrappingKey, ui32KeyLen);
	pucUncommittedLogHdr->ui32DbKeyLen = ui32KeyLen;

	m_pDatabase->m_rcLimitedCode = NE_XFLM_OK;
	m_pDatabase->m_bInLimitedMode = FALSE;
	m_pDatabase->m_bHaveEncKey = TRUE;

	// If we have a dictionary, update its limited mode flag too.

	if (m_pDict)
	{
		m_pDict->m_bInLimitedMode = FALSE;
	}

	// Log the upgrade packet

	pRfl->enableLogging( &uiRflToken);

	if (RC_BAD( rc = pRfl->logEncryptionKey( this, 
		RFL_ENABLE_ENCRYPTION_PACKET, pucWrappingKey, ui32KeyLen)))
	{
		goto Exit;
	}

	// Turn logging off again

	pRfl->disableLogging( &uiRflToken);

	// Commit the transaction and force a checkpoint

	if( RC_BAD( rc = commitTrans( 0, TRUE, NULL)))
	{
		goto Exit;
	}

	bStartedTrans = FALSE;

Exit:

	if( bStartedTrans)
	{
		transAbort();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bLocked)
	{
		dbUnlock();
	}

	if( pucWrappingKey)
	{
		f_free( &pucWrappingKey);
	}

	return( rc);
}

/****************************************************************************
Desc : Change the database key from wrapped in the NICI server
		 key to shrouded in a password and vice-versa.

		 If no password is specified, the key will be wrapped in the
		 NICI server key.  If a password is specified, the key will be
		 shrouded in it.
****************************************************************************/
RCODE XFLAPI F_Db::wrapKey(
	const char *		pszPassword)
{
	RCODE					rc = NE_XFLM_OK;
	XFLM_DB_HDR *		pUncommittedDbHdr = &m_pDatabase->m_uncommittedDbHdr;
	FLMBOOL				bStartedTrans = FALSE;
	FLMBYTE *			pucTmp = NULL;
	FLMUINT32			ui32KeyLen = XFLM_MAX_ENC_KEY_SIZE;
	F_Rfl *				pRfl = m_pDatabase->m_pRfl;
	FLMBOOL				bLocked = FALSE;
	FLMUINT				uiRflToken = 0;
	
	if( getTransType() != XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}

	if( !(m_uiFlags & FDB_HAS_FILE_LOCK))
	{
		if ( RC_BAD( rc = dbLock( FLM_LOCK_EXCLUSIVE, 0, FLM_NO_TIMEOUT)))
		{
			goto Exit;
		}
		bLocked = TRUE;
	}

	// Turn off logging.  We only want to log the wrap key packet.

	pRfl->disableLogging( &uiRflToken);

	// Start the transaction

	if (RC_BAD( rc = transBegin( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	bStartedTrans = TRUE;
	
	// Wrap or shroud the key

	if (RC_BAD( rc = m_pDatabase->m_pWrappingKey->getKeyToStore(
							&pucTmp, &ui32KeyLen,
							(FLMBYTE *)pszPassword, NULL)))
	{
		goto Exit;
	}
	
	f_memcpy( pUncommittedDbHdr->DbKey, pucTmp, ui32KeyLen);
	pUncommittedDbHdr->ui32DbKeyLen = ui32KeyLen;

	// Turn on logging.  We only want to log the wrap key packet.

	pRfl->enableLogging( &uiRflToken);

	// Log a wrapped key packet to record that the key 
	// has been wrapped/encrypted.

	if (RC_BAD( rc = pRfl->logEncryptionKey( this, 
		RFL_WRAP_KEY_PACKET, pucTmp, ui32KeyLen)))
	{
		goto Exit;
	}

	// Turn logging off again

	pRfl->disableLogging( &uiRflToken);

	// Make sure the log header gets written out...

	m_bHadUpdOper = TRUE;
	
	// Commit the transaction and force a checkpoint

	if (RC_BAD( rc = transCommit()))
	{
		goto Exit;
	}

	bStartedTrans = FALSE;

	// Delete the old password
	
	if (m_pDatabase->m_pszDbPasswd)
	{
		f_free( &m_pDatabase->m_pszDbPasswd);
	}

	// Store the new password
	
	if( pszPassword)
	{
		if (RC_BAD( rc = f_calloc( f_strlen( pszPassword) + 1,
			&m_pDatabase->m_pszDbPasswd)))
		{
			goto Exit;
		}
		
		f_memcpy( m_pDatabase->m_pszDbPasswd, pszPassword,
			(FLMSIZET)f_strlen( pszPassword));
	}

Exit:

	if( bStartedTrans)
	{
		transAbort();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bLocked)
	{
		dbUnlock();
	}

	if( pucTmp)
	{
		f_free( &pucTmp);
	}
	
	return( rc);
}

/***************************************************************************
 * A private function that goes through all the potential trial-and-error 
 * involved in getting the strongest possible key we can use for the 
 * database key.
 * NOTE: If an update transaction is needed, it is the responsibility of
 * the caller to start the transaction!  We can't check for this in the
 * function because sometimes a transaction is required, and other times
 * (such as during a db create) it's impossible.
 ***************************************************************************/
RCODE F_Db::createDbKey( void)
{
	RCODE		rc = NE_XFLM_OK;

	if( m_pDatabase->m_pWrappingKey)
	{
		m_pDatabase->m_pWrappingKey->Release();
		m_pDatabase->m_pWrappingKey = NULL;
	}
	
	if( RC_BAD( rc = flmAllocCCS( &m_pDatabase->m_pWrappingKey)))
	{
		goto Exit;
	}

	if( RC_BAD( rc = m_pDatabase->m_pWrappingKey->init( 
		TRUE, FLM_NICI_AES)))
	{
		goto Exit;
	}
	
	// Try to get the strongest encryption supported on this platform.
	
	if( RC_BAD( rc = m_pDatabase->m_pWrappingKey->generateWrappingKey(
		XFLM_NICI_AES256)))
	{
		if( RC_BAD( rc = m_pDatabase->m_pWrappingKey->generateWrappingKey(
			XFLM_NICI_AES192)))
		{
			if( RC_BAD( rc = m_pDatabase->m_pWrappingKey->generateWrappingKey(
				XFLM_NICI_AES128)))
			{
				// Try using DES3
				
				m_pDatabase->m_pWrappingKey->Release();
				m_pDatabase->m_pWrappingKey = NULL;
				
				if( RC_BAD( rc = flmAllocCCS( &m_pDatabase->m_pWrappingKey)))
				{
					goto Exit;
				}
				
				if (RC_BAD( rc = m_pDatabase->m_pWrappingKey->init(
					TRUE, FLM_NICI_DES3)))
				{
					goto Exit;
				}
		
				if (RC_BAD( rc = m_pDatabase->m_pWrappingKey->generateWrappingKey(
					XFLM_NICI_DES3X)))
				{
					// No more choices...
					goto Exit;
				}
			}
		}
	}
	
Exit:

	return rc;
}

/****************************************************************************
Desc : Generate a new database key and re-wrap all existing keys in it
		 NOTE:  New database key will be wrapped in NICI server key,
		 even if the previous key was wrapped in a password.
****************************************************************************/
RCODE XFLAPI F_Db::rollOverDbKey( void)
{
	RCODE					rc = NE_XFLM_OK;
	F_Rfl *				pRfl = m_pDatabase->m_pRfl;
	FLMBYTE *			pucWrappingKey = NULL;
	FLMUINT32			ui32KeyLen = 0;
	FLMUINT				uiEncDefNum;
	F_ENCDEF *			pEncDef;
	F_DOMNode *			pNode = NULL;
	F_DOMNode *			pAttrNode = NULL;
	FLMBYTE *			pucBuf = NULL;
	FLMBOOL				bLocked = FALSE;
	FLMBOOL				bStartedTrans = FALSE;
	FLMUINT32			ui32BufLen = 0;
	FLMUINT				uiRflToken = 0;
	
	if( getTransType() != XFLM_NO_TRANS)
	{
		rc = RC_SET( NE_XFLM_TRANS_ACTIVE);
		goto Exit;
	}

	if (!(m_uiFlags & FDB_HAS_FILE_LOCK))
	{
		if ( RC_BAD( rc = dbLock( FLM_LOCK_EXCLUSIVE, 0, FLM_NO_TIMEOUT)))
		{
			goto Exit;
		}
		
		bLocked = TRUE;
	}

	// Turn off logging

	pRfl->disableLogging( &uiRflToken);

	// Start the transaction

	if (RC_BAD( rc = transBegin( XFLM_UPDATE_TRANS)))
	{
		goto Exit;
	}
	bStartedTrans = TRUE;
	
	// Update the database header with the new key
	
	if( RC_BAD( rc = createDbKey()))
	{
		goto Exit;
	}
	
	if( RC_BAD( rc = m_pDatabase->m_pWrappingKey->getKeyToStore(
		&pucWrappingKey, &ui32KeyLen, 
		m_pDatabase->m_pszDbPasswd, NULL)))
	{
		goto Exit;
	}
	
	f_memcpy( m_pDatabase->m_uncommittedDbHdr.DbKey, 
		pucWrappingKey, ui32KeyLen);
	m_pDatabase->m_uncommittedDbHdr.ui32DbKeyLen = ui32KeyLen;

	// Loop through all the keys in the dictionary
	
	for( uiEncDefNum = m_pDict->m_uiLowestEncDefNum;
			 uiEncDefNum <= m_pDict->m_uiHighestEncDefNum; uiEncDefNum++)
	{
		// If we can't retrieve the encdef, it's no big deal.  Just means
		// there was a gap in the enc def nums.
		
		if( RC_OK( rc = m_pDict->getEncDef( uiEncDefNum, &pEncDef)))
		{
			if( RC_BAD( rc = getNode( XFLM_DICT_COLLECTION, 
				pEncDef->ui64DocumentId, &pNode)))
			{
				goto Exit;
			}
			
			// Get the attribute holding the actual key
			
			if( RC_BAD( rc = pNode->getAttribute( this, 
				ATTR_ENCRYPTION_KEY_TAG, (IF_DOMNode **)&pAttrNode)))
			{
				goto Exit;
			}
			
			if( RC_BAD( rc = pEncDef->pCcs->getKeyToStore( &pucBuf, &ui32BufLen,
				NULL, m_pDatabase->m_pWrappingKey)))
			{
				goto Exit;
			}

			pAttrNode->removeModeFlags( this, FDOM_READ_ONLY);
			if( RC_BAD( rc = pAttrNode->setBinary( this, pucBuf, ui32BufLen)))
			{
				goto Exit;
			}
			pAttrNode->addModeFlags( this, FDOM_READ_ONLY);

			f_free( &pucBuf);
			ui32BufLen = 0;

			if( RC_BAD( rc = documentDone( XFLM_DICT_COLLECTION,
				pEncDef->ui64DocumentId)))
			{
				goto Exit;
			}	
		}
		else
		{
			rc = NE_XFLM_OK;
		}
	}
	
	if( RC_BAD( rc = transCommit()))
	{
		goto Exit;
	}
	bStartedTrans = FALSE;
	
	pRfl->enableLogging( &uiRflToken);
	
	if( RC_BAD( rc = m_pDatabase->m_pRfl->logRollOverDbKey( this)))
	{
		goto Exit;
	}

Exit:

	if( bStartedTrans)
	{
		transAbort();
	}

	if( uiRflToken)
	{
		pRfl->enableLogging( &uiRflToken);
	}

	if( bLocked)
	{
		dbUnlock();
	}
	
	if( pucWrappingKey)
	{
		f_free( &pucWrappingKey);
	}
	
	if( pucBuf)
	{
		f_free( &pucBuf);
	}

	if( pNode)
	{
		pNode->Release();
	}
	
	if( pAttrNode)
	{
		pAttrNode->Release();
	}

	return( rc);
}
