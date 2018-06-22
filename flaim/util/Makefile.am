EXTRA_DIST = ha_flaim.cpp ha_flaim.h rflread.cpp rflread.h viewrfl.cpp

FLAIM_INCLUDE = -I$(top_srcdir)/src
FLAIM_LDADD = ../src/libflaim.la

AM_CPPFLAGS = $(FLAIM_INCLUDE) $(FTK_INCLUDE)
LDADD = libutil.la $(FLAIM_LDADD)

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

bin_PROGRAMS = flmcheckdb flmrebuild flmview flmdbshell flmgigatest

flmcheckdb_SOURCES = checkdb.cpp
flmrebuild_SOURCES = rebuild.cpp
flmgigatest_SOURCES = gigatest.cpp gigatest.h

flmview_SOURCES = \
 flmarg.h \
 viewblk.cpp \
 view.cpp \
 viewdisp.cpp \
 viewedit.cpp \
 viewfhdr.cpp \
 view.h \
 viewlfil.cpp \
 viewlhdr.cpp \
 viewmenu.cpp \
 viewsrch.cpp

flmdbshell_SOURCES = \
 dbshell.cpp \
 dbshell.h \
 flm_edit.cpp \
 flm_edit.h

## Check Programs

check_PROGRAMS = ut_basictest
check_HEADERS = flmunittest.h

ut_basictest_SOURCES = basic_test.cpp flmunittest.cpp

## Unit Tests

TESTS = ut_basictest

clean-local:
	rm -rf samplebackup samplerestore.*

