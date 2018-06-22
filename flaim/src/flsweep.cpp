//-------------------------------------------------------------------------
// Desc:	Sweep database to check field usage.
// Tabs:	3
//
// Copyright (c) 1996-2007 Novell, Inc. All Rights Reserved.
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
Desc:
****************************************************************************/
class DbDict : public F_Object
{
public:

	DbDict()
	{
		m_puiStateTbl = NULL;
		m_bInternalTrans = FALSE;
	}

	~DbDict();

	RCODE init(
		FDB *			pDb,
		FLMBOOL		bInternalTrans,
		FLMUINT		uiMode,
		FLMBOOL *	pbFoundPurgeField);

	FINLINE FLMUINT getState(
		FLMUINT		uiFieldID)
	{
		if( uiFieldID > m_uiTblSize)
		{
			return( 0);
		}
		else
		{
			return( m_puiStateTbl[ uiFieldID]);
		}
	}

	RCODE changeState(
		FLMUINT		uiFieldID,
		FLMUINT		uiNewState);

	RCODE finish( void);

private:

	FDB *				m_pDb;
	FLMBOOL			m_bInternalTrans;
	FLMUINT *		m_puiStateTbl;
	FLMUINT			m_uiTblSize;
};

/****************************************************************************
Desc:
****************************************************************************/
class DbWalk : public F_Object
{
public:

	SWEEP_INFO		m_SwpInfo;

	DbWalk()
	{
		f_memset( &m_SwpInfo, 0, sizeof( SWEEP_INFO));
		m_pDb = NULL;
		m_bInternalTrans = FALSE;
		m_uiCallbackFreq = 0;
		m_fnStatusHook = NULL;
		m_UserData = 0;
		m_uiNextLFile = 0;
		m_uiRecsRead = 0;
		m_uiLastDrn = 0;
	}

	~DbWalk()
	{
	}

	FINLINE void init(
		FDB *			pDb,
		FLMBOOL		bInternalTrans,
		FLMUINT		uiCallbackFreq,
		STATUS_HOOK fnStatusHook,
		void *	  	UserData)
	{
		m_pDb = pDb;
		m_bInternalTrans = bInternalTrans;
		m_uiCallbackFreq = uiCallbackFreq;
		m_fnStatusHook = fnStatusHook;
		m_UserData = UserData;
	}

	RCODE nextContainer(
		FLMUINT *		puiContainer);

	RCODE nextRecord(
		FlmRecord **	ppRecord);

	RCODE updateRecord(
		FLMUINT			uiDrn,
		FlmRecord *		pRecord);

private:

	FDB *				m_pDb;
	FLMBOOL			m_bInternalTrans;
	FLMUINT			m_uiCallbackFreq;
	STATUS_HOOK		m_fnStatusHook;
	void *			m_UserData;
	FLMUINT			m_uiNextLFile;
	FLMUINT			m_uiRecsRead;
	FLMUINT			m_uiLastDrn;
};

/****************************************************************************
Desc:		Provides the ability to scan a FLAIM database for the purpose of
			performing maintenance activities and collecting statistics.
Notes:	During a database sweep, the user may perform one or all of the
			following:

			 1) Check field and record template usage:  When FlmDbSweep finds
			 occurrences of fields or records that have a status of 'checking',
			 their status will be changed to 'active'.  If no occurence of a
			 particular field/template is found during the database sweep, the
			 status of the item will be change from 'checking' to 'unused'.
	
			 Note:  An 'unused' status associated with a field indicates that no
			 occurances of the field were found within any data records. The field
			 may still be referenced from an index or record template definition.
			 It is the user's responsibility to remove any dictionary references
			 to the 'unused' item before attempting to delete it.
	
			 2) Purge field and record templates from a database:  This option
			 will remove any occurances of fields or records that have a 'purge'
			 status.  Before removing occurances, the dictionary is checked to
			 verify that no other dictionary definitions reference the item to be
			 purged.  If references are found, an error is returned.  Otherwise,
			 'purged' items will be deleted.
	
			 3) Visit database items:  This option allows the user to visit
			 all items within the database and gather statistics about the
			 database and it's contents.
****************************************************************************/
FLMEXP RCODE FLMAPI FlmDbSweep(
	HFDB				hDb,
	FLMUINT			uiSweepMode,
	FLMUINT			uiCallbackFreq,
	STATUS_HOOK 	fnStatusHook,
	void *			UserData)
{
	RCODE				rc = FERR_OK;
	DbDict *			pDbDict = NULL;
	DbWalk *			pDbWalk = NULL;
	FlmRecord *		pRecord = NULL;
	FLMUINT			uiContainer;
	FLMUINT			uiState;
	SWEEP_INFO 		SwpInfo;
	FLMBOOL			bStartedTrans = FALSE;
	FDB *				pDb = (FDB *) hDb;
	FLMUINT			uiEncState;

	if( IsInCSMode( pDb))
	{
		fdbInitCS( pDb);
		rc = RC_SET( FERR_NOT_IMPLEMENTED);
		goto ExitCS;
	}

	if( (pDbWalk = f_new DbWalk) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	f_memset( &SwpInfo, 0, sizeof( SWEEP_INFO));
	
	if( RC_BAD( rc = fdbInit( pDb, FLM_NO_TRANS,
		FDB_TRANS_GOING_OK, 0, NULL)))
	{
		goto Exit;
	}
	
	if( pDb->uiTransType == FLM_READ_TRANS)
	{
		rc = RC_SET_AND_ASSERT( FERR_ILLEGAL_TRANS_OP);
		goto Exit;
	}
	else if( pDb->uiTransType == FLM_NO_TRANS)
	{
		if( RC_BAD( rc = flmBeginDbTrans( pDb, FLM_READ_TRANS,
			0, FLM_DONT_POISON_CACHE)))
		{
			goto Exit;
		}

		bStartedTrans = TRUE;
	}
			
	pDbWalk->init( pDb, bStartedTrans, uiCallbackFreq, fnStatusHook, UserData);
	SwpInfo.hDb = pDbWalk->m_SwpInfo.hDb = hDb;

	// Only initialize a DbDict if needed
	
	if( (uiSweepMode & SWEEP_CHECKING_FLDS) || 
		(uiSweepMode & SWEEP_PURGED_FLDS))
	{
		FLMBOOL		bFoundPurgeField;

		if( (pDbDict = f_new DbDict) == NULL)
		{
			rc = RC_SET( FERR_MEM);
			goto Exit;
		}

		if( RC_BAD( rc = pDbDict->init( pDb, bStartedTrans, 
			uiSweepMode, &bFoundPurgeField)))
		{
			goto Exit;
		}

		// If user is performing purge field sweep and dictionary contains no
		// purged fields then just return.

		if( uiSweepMode == SWEEP_PURGED_FLDS && !bFoundPurgeField)
		{
			goto Exit;
		}
	}

	for (;;)
	{
		// Get the next container

		if( RC_BAD( rc = pDbWalk->nextContainer( &uiContainer)))
		{
			if( rc == FERR_EOF_HIT)
			{
				// No more containers to process.
				
				rc = FERR_OK;
				break;
			}
			else
			{
				goto Exit;
			}
		}

		if( !pDbDict &&
			 !(uiCallbackFreq & EACH_RECORD || uiCallbackFreq & EACH_FIELD))
		{
			// User is performing a status sweep and they are not visiting
			// each record or field.  Go to next container.
			
			continue;
		}

		SwpInfo.uiContainer = uiContainer;

		// Visit each record in the container.

		for( ;;)
		{
			FLMBOOL		bRecChanged;
			void *		pvField;
			void *		pvPrevField;
			FLMUINT		uiDrn;

			if( RC_BAD( rc = pDbWalk->nextRecord( &pRecord)))
			{
				if( rc == FERR_EOF_HIT)
				{
					rc = FERR_OK;
					break;
				}
				else
				{
					goto Exit;
				}
			}

			bRecChanged = FALSE;
			uiDrn = pRecord->getID();
			SwpInfo.uiRecId = uiDrn;

			for( pvField = pRecord->root(); pvField; 
				pvField = pRecord->next( pvField))
			{
				SwpInfo.pRecord = pRecord;
				SwpInfo.pvField = pvField;

				// Call the field call-back

				if( (uiCallbackFreq & EACH_FIELD) && fnStatusHook)
				{
					if( RC_BAD( rc = (fnStatusHook)( FLM_SWEEP_STATUS,
							(void *)&SwpInfo, (void *)EACH_FIELD, UserData)))
					{
						if( rc == FERR_EOF_HIT)
						{
							// User returned FERR_EOF_HIT, means skip to next record.

							rc = FERR_OK;
							break;
						}
						else
						{
							goto Exit;
						}
					}
				}

				// Continue to next field if no DbDict object.

				if( !pDbDict)
				{
					continue;
				}

				uiState = pDbDict->getState( pRecord->getFieldID( pvField));
				
				// If the field is encrypted, we need to check the state of the 
				// encryption definition record as well, since it is being
				// referenced by this field.
				
				if( pRecord->isEncryptedField( pvField))
				{
					uiEncState = pDbDict->getState( 
											pRecord->getEncryptionID( pvField));
				}
				else
				{
					uiEncState = 0;
				}
				
				if( !uiState && !uiEncState)
				{
					continue;
				}

				if( (uiState == ITT_FLD_STATE_CHECKING ||
					  uiState == ITT_FLD_STATE_PURGE ||
					  uiEncState == ITT_ENC_STATE_CHECKING ||
					  uiEncState == ITT_ENC_STATE_PURGE) &&
					 (uiCallbackFreq & EACH_CHANGE) && fnStatusHook)
				{
					if( RC_BAD( rc = (fnStatusHook)( FLM_SWEEP_STATUS,
						(void *)&SwpInfo, (void *)EACH_CHANGE, UserData)))
					{
						goto Exit;
					}
				}

				if( uiState == ITT_FLD_STATE_CHECKING)
				{
					// Change the field's state to 'active'

					if( RC_BAD( rc = pDbDict->changeState(
						pRecord->getFieldID( pvField), ITT_FLD_STATE_ACTIVE)))
					{
						goto Exit;
					}
				}
				else if( uiState == ITT_FLD_STATE_PURGE)
				{
					// If needed, create writeable version of record and
					// reposition to field

					if( pRecord->isReadOnly())
					{
						FlmRecord *	pTmpRec;
						FLMUINT		uiFieldID = pRecord->getFieldID( pvField);
						
						if( (pTmpRec = pRecord->copy()) == NULL)
						{
							rc = RC_SET( FERR_MEM);
							goto Exit;
						}
						
						pRecord->Release();
						pRecord = pTmpRec;

						pvField = pRecord->find( pRecord->root(), uiFieldID);

						// Should always be able to re-find the field.

						flmAssert( pvField);
					}
					
					// Remove the purged field from the record.

					bRecChanged = TRUE;
					if( pvField == pRecord->root())
					{
						// Passing a NULL pRecord to updateRecord will delete
						// the record.

						pRecord->Release();
						pRecord = NULL;

						// Get out of the for loop - nothing more to do with
						// this record

						break;
					}
					else
					{
						// Must save the previous field before removing the
						// field so we can set pvField to it after removing
						// pvField.

						pvPrevField = pRecord->prev( pvField);
						pRecord->remove( pvField);
						pvField = pvPrevField;
					}
				}

				// Check the EncDef state, independant of the field state.  If the
				// field has been purged, we don't need to update the EncDef
				// record since it is no longer being referenced by this field.
				
				if( uiEncState == ITT_ENC_STATE_CHECKING && 
					 uiState != ITT_FLD_STATE_PURGE)
				{
					// Change the EncDef record's state to 'active'
	
					if( RC_BAD( rc = pDbDict->changeState(
						pRecord->getEncryptionID( pvField), ITT_ENC_STATE_ACTIVE)))
					{
						goto Exit;
					}
				}
				else if( uiEncState == ITT_ENC_STATE_PURGE &&
							uiState != ITT_FLD_STATE_PURGE)
				{
					FLMUINT				uiDataLength = pRecord->getDataLength( pvField);
					const FLMBYTE *	pucDataSource = pRecord->getDataPtr( pvField);
					FLMBYTE *			pucDestPtr;
					
					// If the EncDef record has a state of purge, then we must
					// change the field of this record so that it is no longer
					// encrypted.
					
					if( RC_BAD( rc = pRecord->allocStorageSpace( pvField,
						pRecord->getDataType( pvField), uiDataLength, 0, 0, 0,
						&pucDestPtr, NULL)))
					{
						goto Exit;
					}
					
					f_memmove(pucDestPtr, pucDataSource, uiDataLength);
					bRecChanged = TRUE;
				}
			}	

			// Record was changed because a purged field was found.

			if( bRecChanged)
			{
				if( RC_BAD( rc = pDbWalk->updateRecord( uiDrn, pRecord)))
				{
					goto Exit;
				}
			}
		}
	}

	// Now complete any changes needed within the dictionary

	if( pDbDict)
	{
		rc = pDbDict->finish();
	}

Exit:

	if( pDbWalk)
	{
		pDbWalk->Release();
		pDbWalk = NULL;	
	}
	
	if( pDbDict)			
	{
		pDbDict->Release();
		pDbWalk = NULL;
	}

	if( bStartedTrans)
	{
		if(  pDb->uiTransType != FLM_NO_TRANS)
		{
			(void)flmAbortDbTrans( pDb);
		}
	}

ExitCS:

	flmExit( FLM_DB_SWEEP, pDb, rc);

	if( pRecord)
	{
		pRecord->Release();
	}
	
	return( rc);
}

/****************************************************************************
Desc:	Get the next container within this database
****************************************************************************/
RCODE DbWalk::nextContainer(
	FLMUINT *	puiContainer)
{
	LFILE *		pLFileTbl = (LFILE *)m_pDb->pDict->pLFileTbl;
	FLMBOOL		bFoundContainer = FALSE;
	RCODE			rc = FERR_OK;
	FLMUINT		uiTblSize;

	for( uiTblSize = m_pDb->pDict->uiLFileCnt;
		  m_uiNextLFile < uiTblSize;
		  m_uiNextLFile++)
	{
		if( pLFileTbl [m_uiNextLFile].uiLfType == LF_CONTAINER &&
			 (pLFileTbl [m_uiNextLFile].uiLfNum == FLM_DATA_CONTAINER ||
			  pLFileTbl [m_uiNextLFile].uiLfNum < FLM_DICT_CONTAINER))
		{
			// We've found the next container to visit.

			m_SwpInfo.uiContainer = *puiContainer =
			pLFileTbl [m_uiNextLFile].uiLfNum;

			// Note: don't need to release pRecord, because an addRef wasn't done.

			m_SwpInfo.pRecord = NULL;
			m_SwpInfo.pvField = NULL;

			if( (m_uiCallbackFreq & EACH_CONTAINER) && m_fnStatusHook)
			{
				if( RC_BAD( rc = (m_fnStatusHook)( FLM_SWEEP_STATUS,
					(void *)&m_SwpInfo, (void *)EACH_CONTAINER, m_UserData)))
				{
					if( rc == FERR_EOF_HIT)
					{

						// User wants to skip this container.

						continue;
					}
					else
					{
						goto Exit;
					}
				}
			}

			// Perform needed setup to read this containers records.

			m_uiRecsRead = 0;
			m_uiLastDrn = 0;
			m_uiNextLFile++;
			bFoundContainer = TRUE;
			break;
		}
	}

	if( !bFoundContainer)
	{
		rc = RC_SET( FERR_EOF_HIT);
		goto Exit;
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Returns the next record within the current container.
****************************************************************************/
RCODE DbWalk::nextRecord(
	FlmRecord **	ppRecord)
{
	RCODE				rc = FERR_OK;
	FLMUINT			uiDrn;

	// Loop till we get a record or hit end of container.

	for( ;;)
	{
		if( m_bInternalTrans)
		{
			// Abort and start a new read transaction every 1000 records.
	
			if( (m_uiRecsRead % 1000) == 0)
			{
AbortTrans:
	
				if( m_pDb->uiTransType != FLM_NO_TRANS)
				{
					(void)flmAbortDbTrans( m_pDb);
				}
	
				if( RC_BAD( rc = flmBeginDbTrans( m_pDb, FLM_READ_TRANS,
					0, FLM_DONT_POISON_CACHE)))
				{
					goto Exit;
				}
			}
		}

		if( *ppRecord)
		{
			(*ppRecord)->Release();
			*ppRecord = NULL;
		}

		if( RC_BAD( rc = FlmRecordRetrieve( (HFDB)m_pDb, m_SwpInfo.uiContainer,
			m_uiLastDrn, FO_EXCL, ppRecord, &uiDrn)))
		{
			if( rc == FERR_OLD_VIEW)
			{
				if( m_bInternalTrans)
				{
					goto AbortTrans;
				}
				
				goto Exit;
			}

			// It is possible for the container to go away in the
			// middle of walking through it, because we are stopping
			// and starting transactions.

			if( rc == FERR_BAD_CONTAINER)
			{
				// Must abort the transaction because FERR_BAD_CONTAINER
				// will not allow the transaction to continue
				// if we are in an update transaction.
				
				if( !m_bInternalTrans)
				{
					goto Exit;
				}
				
				if( m_pDb->uiTransType != FLM_NO_TRANS)
				{
					(void)flmAbortDbTrans( m_pDb);
				}
				
				if( RC_BAD( rc = flmBeginDbTrans( m_pDb, FLM_READ_TRANS,
					0, FLM_DONT_POISON_CACHE)))
				{
					goto Exit;
				}

				// Change error code so that it looks like we have
				// hit the end of the container.

				rc = RC_SET( FERR_EOF_HIT);
			}
			
			goto Exit;
		}
		
		m_uiLastDrn = uiDrn;
		m_uiRecsRead++;
		m_SwpInfo.pRecord = *ppRecord;
		m_SwpInfo.pvField = NULL;

		// Make STATUS_HOOK callback

		if( (m_uiCallbackFreq & EACH_RECORD) && m_fnStatusHook)
		{
			if( (rc = (m_fnStatusHook)( FLM_SWEEP_STATUS, (void *)&m_SwpInfo,
				(void *) EACH_RECORD, m_UserData)) == FERR_EOF_HIT)
			{
				// User wants to skip this record.
				
				continue;
			}
			else
			{
				// Return this record
				
				break;
			}
		}
		else
		{
			// Return the record
			
			break;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	The last record returned from DbWalk::nextRecord contained 'purged' 
		fields which have been removed. This function will now replace the 
		old record with the new record. This record may be == NULL, in which
		case the record will be deleted.
****************************************************************************/
RCODE DbWalk::updateRecord(
	FLMUINT			uiDrn,
	FlmRecord *		pRecord)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bRestartTrans = FALSE;

	if( m_bInternalTrans)
	{
		if( m_pDb->uiTransType != FLM_NO_TRANS)
		{
			(void)flmAbortDbTrans( m_pDb);
			bRestartTrans = TRUE;
		}
	}

	// Either modify or delete the specified record

	rc = ( pRecord)
			? FlmRecordModify( m_pDb, m_SwpInfo.uiContainer,
										uiDrn, pRecord, FLM_AUTO_TRANS | FLM_NO_TIMEOUT)
			: FlmRecordDelete( m_pDb,	m_SwpInfo.uiContainer,
										uiDrn, FLM_AUTO_TRANS | FLM_NO_TIMEOUT);
	if( RC_BAD( rc))
	{
		goto Exit;
	}

	// Now (if needed) restart the read transaction.

	if( bRestartTrans)
	{
		flmAssert( m_bInternalTrans);
		
		if( RC_BAD( rc = flmBeginDbTrans( m_pDb, FLM_READ_TRANS,
								0, FLM_DONT_POISON_CACHE)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}


/****************************************************************************
Desc:
****************************************************************************/
DbDict::~DbDict()
{
	if( m_pDb)
	{
		m_pDb->bFldStateUpdOk = FALSE;
	}

	if( m_puiStateTbl)
	{
		f_free( &m_puiStateTbl);
	}
}

/****************************************************************************
Desc:		Read the dictionary and create a internal table that records
			the state of field templates within the specified dictionary
****************************************************************************/
RCODE DbDict::init(
	FDB *			pDb,
	FLMBOOL		bInternalTrans,
	FLMUINT		uiMode,
	FLMBOOL *	pbFoundPurgeField)
{
	RCODE			rc = FERR_OK;
	ITT *			pItt = NULL;
	FLMUINT		uiItem;
	FLMUINT		uiCount;
	FLMUINT		uiStateMask = 0;

	*pbFoundPurgeField = FALSE;
	m_pDb = pDb;
	m_bInternalTrans = bInternalTrans;

	// Need to set a flag that which tells lower level FLAIM code
	//	that its okay to:
	//		1) Set a field state to 'unused' and
	//		2) Delete a field thats in a 'purged' state.
	
	m_pDb->bFldStateUpdOk = TRUE;

	// Allocate state table
	
	m_uiTblSize = m_pDb->pDict->uiIttCnt;
	
	if( RC_BAD( rc = f_calloc( 
		(m_uiTblSize * sizeof( FLMUINT)), &m_puiStateTbl)))
	{
		goto Exit;
	}

	uiCount = m_pDb->pDict->uiIttCnt;
	pItt = m_pDb->pDict->pIttTbl;

	if( uiMode & SWEEP_CHECKING_FLDS)
	{
		uiStateMask |= ITT_FLD_STATE_CHECKING;
	}

	if( uiMode & SWEEP_PURGED_FLDS)
	{
		uiStateMask |= ITT_FLD_STATE_PURGE;
	}

	// Now loop through the ITT table and set the correct states within
	// the DbDict's state table
	
	for( uiItem = 0; uiItem < uiCount; pItt++, uiItem++)
	{
		// Make sure the entry is a field or an encryption definition record
		
		if( ITT_IS_FIELD( pItt))
		{
			m_puiStateTbl[ uiItem] = (pItt->uiType & uiStateMask);
			if( m_puiStateTbl[ uiItem] == ITT_FLD_STATE_PURGE)
			{
				*pbFoundPurgeField = TRUE;

				// Make sure this field is not references and
				// can therefore be deleted
				
				if( RC_BAD( rc = flmCheckDictFldRefs( m_pDb->pDict, uiItem)))
				{
					goto Exit;
				}
			}
		}
		else if( ITT_IS_ENCDEF( pItt) && !m_pDb->pFile->bInLimitedMode)
		{
			if( RC_BAD( rc = fdictGetEncInfo( m_pDb, uiItem, NULL,
				&m_puiStateTbl[ uiItem])))
			{
				goto Exit;
			}
			
			if( m_puiStateTbl[ uiItem] == ITT_ENC_STATE_PURGE)
			{
				*pbFoundPurgeField = TRUE;

				// Make sure this field is not references and
				// can therefore be deleted.
				
				if( RC_BAD( rc = flmCheckDictEncDefRefs( m_pDb->pDict, uiItem)))
				{
					goto Exit;
				}
			}
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Change the specified field's state.
		Currently this is only for "Checking" -> "Active"
****************************************************************************/
RCODE DbDict::changeState(
	FLMUINT		uiFieldID,
	FLMUINT		uiNewState)
{
	RCODE			rc = FERR_OK;
	FLMBOOL		bRestartTrans = FALSE;

	if( m_puiStateTbl[ uiFieldID] != ITT_FLD_STATE_CHECKING)
	{
		return( RC_SET( FERR_FAILURE));
	}

	if( m_bInternalTrans)
	{
		if( m_pDb->uiTransType != FLM_NO_TRANS)
		{
			(void)flmAbortDbTrans( m_pDb);
			bRestartTrans = TRUE;
		}
	}

	// Change the state tables value for this field/record template

	m_puiStateTbl[ uiFieldID] = 0;

	// Change the state of the dictionary item.

	if( RC_BAD( rc = flmChangeItemState( m_pDb, uiFieldID, uiNewState)))
	{
		goto Exit;
	}

	// Now restart the read transaction.
	
	if( bRestartTrans)
	{
		flmAssert( m_bInternalTrans);
		
		if( RC_BAD( rc = flmBeginDbTrans( m_pDb, FLM_READ_TRANS,
								0, FLM_DONT_POISON_CACHE)))
		{
			goto Exit;
		}
	}

Exit:

	return( rc);
}

/****************************************************************************
Desc:	Following a complete sweep of a database, this function will
		handle items that are still marked 'checking' or 'purge'.
****************************************************************************/
RCODE DbDict::finish( void)
{
	RCODE			rc = FERR_OK;
	FLMUINT		uiItem;

	// If we have an internal read transaction going then end it.
	
	if( m_bInternalTrans)
	{
		if( m_pDb->uiTransType != FLM_NO_TRANS)
		{
			(void)flmAbortDbTrans( m_pDb);
		}
	}

	// Loop through the state table changing: 'checking' fields to 'unused' and
	// deleting 'purged' fields

	for( uiItem = 1; uiItem < m_uiTblSize && RC_OK( rc); uiItem++)
	{
		if( m_puiStateTbl[ uiItem] == ITT_FLD_STATE_CHECKING)
		{
			// Change state to "unused"
			
			rc = flmChangeItemState( m_pDb, uiItem, ITT_FLD_STATE_UNUSED);
		}
		else if( m_puiStateTbl[ uiItem] == ITT_FLD_STATE_PURGE)
		{
			// Delete the 'purged' item

			if( RC_BAD( rc = FlmRecordDelete( (HFDB)m_pDb, FLM_DICT_CONTAINER, 
				uiItem, FLM_NO_TIMEOUT | FLM_AUTO_TRANS)))
			{
				goto Exit;
			}
		}
	}

Exit:

	if( m_bInternalTrans)
	{
		if( m_pDb->uiTransType != FLM_NO_TRANS)
		{
			(void)flmAbortDbTrans( m_pDb);
		}
	}

	return( rc);
}

/****************************************************************************
Desc: Change a field/record's defined state. Currently the only supported
		changes are:
			'checking' -> 'active'
			'checking' -> 'unused'
Ret:
****************************************************************************/
RCODE flmChangeItemState(
	FDB *				pDb,
	FLMUINT			uiItemId,
	FLMUINT			uiNewState)
{
	RCODE				rc = FERR_OK;
	FLMBOOL			bStartedTrans = FALSE;
	FlmRecord * 	pRecord = NULL;
	FlmRecord * 	pOldRecord = NULL;
	void *			pvField;

	// If needed, start a update transaction
	
	if( pDb->uiTransType == FLM_NO_TRANS)
	{
		if( RC_BAD( rc = flmBeginDbTrans( pDb, FLM_UPDATE_TRANS, 
			FLM_NO_TIMEOUT, FLM_DONT_POISON_CACHE)))
		{
			goto Exit;
		}
		
		bStartedTrans = TRUE;
	}

	// Now read the dictionary definition

	if( RC_BAD( rc = FlmRecordRetrieve( (HFDB)pDb, FLM_DICT_CONTAINER, 
		uiItemId, FO_EXACT, &pOldRecord, NULL)))
	{
		goto Exit;
	}

	if( (pRecord = pOldRecord->copy()) == NULL)
	{
		rc = RC_SET( FERR_MEM);
		goto Exit;
	}

	// Change the state to the correct state

	pvField = pRecord->find( pRecord->root(), FLM_STATE_TAG);
	flmAssert( pvField != NULL);

	if( RC_BAD( rc = pRecord->setNative( pvField, 
										(uiNewState == ITT_FLD_STATE_UNUSED)
												? "unused"
												: "active")))
	{
		goto Exit;
	}

	// Update the dictionary
	
	if( RC_BAD( rc = FlmRecordModify( (HFDB)pDb, FLM_DICT_CONTAINER, 
		pOldRecord->getID(), pRecord, 0)))
	{
		goto Exit;
	}

Exit:

	if( pRecord)
	{
		pRecord->Release();
	}

	if( pOldRecord)
	{
		pOldRecord->Release();
	}

	if( bStartedTrans)
	{
		if( RC_OK( rc))
		{
			rc = flmCommitDbTrans( pDb, 0, FALSE);
		}
		else
		{
			(void)flmAbortDbTrans( pDb);
		}
	}

	return( rc);
}
