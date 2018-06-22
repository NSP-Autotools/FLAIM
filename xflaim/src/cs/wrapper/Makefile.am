EXTRA_DIST = xflaim cstest sample xflaim.ndoc

xfcs_sources = \
 xflaim/BackupClient.cs \
 xflaim/Backup.cs \
 xflaim/BackupStatus.cs \
 xflaim/CacheInfo.cs \
 xflaim/DataVector.cs \
 xflaim/dbCheck.cs \
 xflaim/DbCopyStatus.cs \
 xflaim/Db.cs \
 xflaim/DbInfo.cs \
 xflaim/dbRebuild.cs \
 xflaim/DbRenameStatus.cs \
 xflaim/DbSystem.cs \
 xflaim/DbSystemStats.cs \
 xflaim/DOMNode.cs \
 xflaim/IStream.cs \
 xflaim/OStream.cs \
 xflaim/Query.cs \
 xflaim/RCODE.cs \
 xflaim/RestoreClient.cs \
 xflaim/RestoreStatus.cs

cstest_sources = \
 cstest/BackupDbTest.cs \
 cstest/CacheTests.cs \
 cstest/CheckDbTest.cs \
 cstest/CompareStringsTests.cs \
 cstest/CopyDbTest.cs \
 cstest/CreateDbTest.cs \
 cstest/cstest.cs \
 cstest/DOMNodeTest.cs \
 cstest/ImportTests.cs \
 cstest/OpenDbTest.cs \
 cstest/RebuildDbTest.cs \
 cstest/RemoveDbTest.cs \
 cstest/RenameDbTest.cs \
 cstest/RestoreDbTest.cs \
 cstest/SettingsTests.cs \
 cstest/StatsTests.cs \
 cstest/StreamTests.cs \
 cstest/VectorTests.cs

TESTS = cstest_script

AM_CSCFLAGS = -d:mono -nologo -warn:4 -warnaserror+ -optimize+
#AM_CSCFLAGS += /debug+ /debug:full /define:FLM_DEBUG

all-local: xflaim_csharp.dll

clean-local:
	rm -f xflaim_csharp.dll xflaim_csharp.xml cstest_script cstest.exe libxflaim.so
	rm -f Output_Stream 
	rm -rf abc backup test.*
	
install-exec-local:
	test -z "$(libdir)" || $(MKDIR_P) "$(DESTDIR)$(libdir)"
	$(INSTALL_PROGRAM) xflaim_csharp.dll "$(DESTDIR)$(libdir)"

install-data-local:
	test -z "$(docdir)" || $(MKDIR_P) "$(DESTDIR)$(docdir)"
	$(INSTALL_DATA) xflaim_csharp.xml "$(DESTDIR)$(docdir)"

uninstall-local:
	rm -f "$(DESTDIR)$(libdir)/xflaim_csharp.dll"
	rm -f "$(DESTDIR)$(docdir)/xflaim_csharp.xml"

xflaim_csharp.dll: $(xfcs_sources)
	@list1='$(xfcs_sources)'; list2=; if test -n "$$list1"; then \
	  for p in $$list1; do \
	    if test -f $$p; then d=; else d="$(srcdir)/"; fi; \
	    list2="$$list2 $$d$$p"; \
	  done; \
	  echo '$(CSC) -target:library $(AM_CSCFLAGS) $(CSCFLAGS) -out:$@ -doc:$(@:.dll=.xml) '"$$list2"; \
	  $(CSC) -target:library $(AM_CSCFLAGS) $(CSCFLAGS) -out:$@ -doc:$(@:.dll=.xml) $$list2; \
	else :; fi

check_SCRIPTS = cstest.exe cstest_script

cstest.exe: xflaim_csharp.dll $(cstest_sources)
	@list1='$(cstest_sources)'; list2=; if test -n "$$list1"; then \
	  for p in $$list1; do \
	    if test -f $$p; then d=; else d="$(srcdir)/"; fi; \
	    list2="$$list2 $$d$$p"; \
	  done; \
	  echo '$(CSC) $(AM_CSCFLAGS) $(CSCFLAGS) -out:$@ '"$$list2"' -reference:xflaim_csharp.dll'; \
	  $(CSC) $(AM_CSCFLAGS) $(CSCFLAGS) -out:$@ $$list2 -reference:xflaim_csharp.dll; \
	else :; fi

cstest_script: cstest.exe
	echo "#!/bin/sh" > cstest_script
	echo "$(top_builddir)/libtool --mode=execute -dlopen ../../libxflaim.la $(CSVM) cstest.exe" >> cstest_script
	chmod 0755 cstest_script

