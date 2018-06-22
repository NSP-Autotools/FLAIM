EXTRA_DIST = dbdiff.cpp dbdiff.h domedit.cpp diffbackups.cpp xmlfiles

XFLAIM_INCLUDE = -I$(top_srcdir)/src
XFLAIM_LDADD = ../src/libxflaim.la

AM_CPPFLAGS = $(XFLAIM_INCLUDE) $(FTK_INCLUDE)
LDADD = libutil.la $(XFLAIM_LDADD)

## Utility Convenience Library 

noinst_LTLIBRARIES = libutil.la

libutil_la_SOURCES = \
 flm_dlst.cpp \
 flm_dlst.h \
 flm_lutl.cpp \
 flm_lutl.h \
 sharutil.cpp \
 sharutil.h

## Utility Programs

bin_PROGRAMS = xflmcheckdb xflmrebuild xflmview xflmdbshell

xflmcheckdb_SOURCES = checkdb.cpp
xflmrebuild_SOURCES = rebuild.cpp

xflmview_SOURCES = \
 viewblk.cpp \
 view.cpp \
 viewdisp.cpp \
 viewedit.cpp \
 viewhdr.cpp \
 view.h \
 viewlfil.cpp \
 viewmenu.cpp \
 viewsrch.cpp

xflmdbshell_SOURCES = \
 domedit.h \
 fdomedt.cpp \
 fshell.cpp \
 fshell.h \
 xshell.cpp

## Check Programs

check_PROGRAMS = \
 ut_basictest \
 ut_binarytest \
 ut_colldeftest \
 ut_dictchangetest \
 ut_dictdeftest \
 ut_dirtyexittest \
 ut_dirtyexittest2 \
 ut_domnodetest \
 ut_enctest \
 ut_importtest \
 ut_indexdeftest \
 ut_indextest \
 ut_indextest2 \
 ut_indextest3 \
 ut_metaphonetest \
 ut_namespacetest \
 ut_regressiontest \
 ut_rfltest \
 ut_sortkeytest \
 ut_sortkeytest2 \
 ut_xpathtest \
 ut_xpathtest2

check_DATA = copy-xml-files.stamp
check_HEADERS = flmunittest.h

ut_basictest_SOURCES = flmunittest.cpp basictestsrv.cpp
ut_binarytest_SOURCES = flmunittest.cpp binarytest.cpp
ut_colldeftest_SOURCES = flmunittest.cpp colldeftestsrv.cpp
ut_dictchangetest_SOURCES = flmunittest.cpp dictchangetest.cpp
ut_dictdeftest_SOURCES = flmunittest.cpp dictdeftestsrv.cpp
ut_dirtyexittest_SOURCES = flmunittest.cpp dirtyexittest1srv.cpp
ut_dirtyexittest2_SOURCES = flmunittest.cpp dirtyexittest2srv.cpp
ut_domnodetest_SOURCES = flmunittest.cpp domnodetestsrv.cpp
ut_enctest_SOURCES = flmunittest.cpp enctestsrv.cpp
ut_importtest_SOURCES = flmunittest.cpp importtestsrv.cpp
ut_indexdeftest_SOURCES = flmunittest.cpp indexdeftestsrv.cpp
ut_indextest_SOURCES = flmunittest.cpp indextest1srv.cpp
ut_indextest2_SOURCES = flmunittest.cpp indextest2srv.cpp
ut_indextest3_SOURCES = flmunittest.cpp indextest3.cpp
ut_metaphonetest_SOURCES = flmunittest.cpp metaphonetestsrv.cpp
ut_namespacetest_SOURCES = flmunittest.cpp namespacetestsrv.cpp
ut_regressiontest_SOURCES = flmunittest.cpp regressiontest.cpp
ut_rfltest_SOURCES = flmunittest.cpp rfltestsrv.cpp
ut_sortkeytest_SOURCES = flmunittest.cpp sortkeytest.cpp
ut_sortkeytest2_SOURCES = flmunittest.cpp sortkeytest2.cpp
ut_xpathtest_SOURCES = flmunittest.cpp xpathtest1srv.cpp
ut_xpathtest2_SOURCES = flmunittest.cpp xpathtest2srv.cpp

## Unit Tests

TESTS = \
 ut_basictest \
 ut_binarytest \
 ut_colldeftest \
 ut_dictchangetest \
 ut_dictdeftest \
 ut_dirtyexittest \
 ut_dirtyexittest2 \
 ut_domnodetest \
 ut_enctest \
 ut_importtest \
 ut_indexdeftest \
 ut_indextest \
 ut_indextest2 \
 ut_indextest3 \
 ut_metaphonetest \
 ut_namespacetest \
 ut_regressiontest \
 ut_rfltest \
 ut_sortkeytest \
 ut_sortkeytest2 \
 ut_xpathtest \
 ut_xpathtest2

## Miscellaneous rules required by Check Programs

copy-xml-files.stamp:
	cp $(srcdir)/xmlfiles/*.xml .
	echo Timestamp > $@

clean-local: 
	rm -rf ix2.*
	rm -rf bld.*
	rm -rf tst.bak
	rm -f *.xml
	rm -f copy-xml-files.stamp

