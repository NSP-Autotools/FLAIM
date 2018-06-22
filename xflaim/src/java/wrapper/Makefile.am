JAVAROOT = .

jarfile = $(PACKAGE)jni-$(VERSION).jar
jardir = $(datadir)/java
pkgpath = xflaim
jhdrout = ..

$(jarfile): $(dist_noinst_JAVA) 
	$(JAR) cvf $(JARFLAGS) $@ $(pkgpath)/*.class

jar_DATA = $(jarfile)

java-headers.stamp: $(classdist_noinst.stamp)
	@list=`echo $(dist_noinst_JAVA) | sed -e 's|\.java||g' -e 's|/|.|g'`; \
	echo "$(JAVAH) -cp . -jni -d $(jhdrout) $(JAVAHFLAGS) $$list"; \
	$(JAVAH) -cp . -jni -d $(jhdrout) $(JAVAHFLAGS) $$list
	@echo "JNI headers generated" > java-headers.stamp

all-local: java-headers.stamp

CLEANFILES = $(jarfile) $(pkgpath)/*.class java-headers.stamp $(jhdrout)/xflaim_*.h

dist_noinst_JAVA = \
 $(pkgpath)/BackupClient.java \
 $(pkgpath)/Backup.java \
 $(pkgpath)/BackupStatus.java \
 $(pkgpath)/BlockIOStats.java \
 $(pkgpath)/CacheInfo.java \
 $(pkgpath)/CacheUsage.java \
 $(pkgpath)/CHECKINFO.java \
 $(pkgpath)/CheckpointInfo.java \
 $(pkgpath)/Collections.java \
 $(pkgpath)/CommitClient.java \
 $(pkgpath)/CompareRules.java \
 $(pkgpath)/CORRUPTINFO.java \
 $(pkgpath)/CountTimeStat.java \
 $(pkgpath)/CREATEOPTS.java \
 $(pkgpath)/DataVector.java \
 $(pkgpath)/DbCheckFlags.java \
 $(pkgpath)/DbCheckStatus.java \
 $(pkgpath)/DbCopyStatus.java \
 $(pkgpath)/DbInfo.java \
 $(pkgpath)/Db.java \
 $(pkgpath)/DbRebuildStatus.java \
 $(pkgpath)/DbRenameStatus.java \
 $(pkgpath)/DbStats.java \
 $(pkgpath)/DbSystem.java \
 $(pkgpath)/DefaultBackupClient.java \
 $(pkgpath)/DefaultRestoreClient.java \
 $(pkgpath)/DeleteStatus.java \
 $(pkgpath)/DictType.java \
 $(pkgpath)/DiskIOStat.java \
 $(pkgpath)/DOMNode.java \
 $(pkgpath)/ExportFormatType.java \
 $(pkgpath)/FlmBackupType.java \
 $(pkgpath)/FlmDataType.java \
 $(pkgpath)/FlmDictIndex.java \
 $(pkgpath)/FlmDomNodeType.java \
 $(pkgpath)/FlmInsertLoc.java \
 $(pkgpath)/ImportStats.java \
 $(pkgpath)/IndexState.java \
 $(pkgpath)/IndexStatus.java \
 $(pkgpath)/InsertLoc.java \
 $(pkgpath)/IStream.java \
 $(pkgpath)/IxClient.java \
 $(pkgpath)/IxStatus.java \
 $(pkgpath)/KeyRetrieveFlags.java \
 $(pkgpath)/LFileStats.java \
 $(pkgpath)/LockStats.java \
 $(pkgpath)/LockType.java \
 $(pkgpath)/LockUser.java \
 $(pkgpath)/OptInfo.java \
 $(pkgpath)/OptType.java \
 $(pkgpath)/OStream.java \
 $(pkgpath)/Query.java \
 $(pkgpath)/QueryOperators.java \
 $(pkgpath)/RCODE.java \
 $(pkgpath)/RebuildStatus.java \
 $(pkgpath)/ReserveID.java \
 $(pkgpath)/RestoreAction.java \
 $(pkgpath)/RestoreClient.java \
 $(pkgpath)/RestoreStatus.java \
 $(pkgpath)/ResultSetCounts.java \
 $(pkgpath)/RTransStats.java \
 $(pkgpath)/SearchFlags.java \
 $(pkgpath)/SlabUsage.java \
 $(pkgpath)/Stats.java \
 $(pkgpath)/TransactionFlags.java \
 $(pkgpath)/TransactionType.java \
 $(pkgpath)/UTransStats.java \
 $(pkgpath)/XFlaimException.java \
 $(pkgpath)/XPathAxis.java

