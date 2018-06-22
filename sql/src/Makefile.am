lib_LTLIBRARIES = libflaimsql.la
include_HEADERS = flaimsql.h

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libflaimsql.pc

libflaimsql_la_SOURCES = \
 btreeinfo.cpp \
 checksum.cpp \
 createdatabase.cpp \
 createindex.cpp \
 createtable.cpp \
 deleterow.cpp \
 dropdatabase.cpp \
 dropindex.cpp \
 droptable.cpp \
 f_btpool.cpp \
 f_btpool.h \
 f_btree.cpp \
 f_btree.h \
 fbtrset.cpp \
 fbtrset.h \
 fcache.h \
 fcollate.cpp \
 fcollate.h \
 fdbcnfig.cpp \
 fdbcopy.cpp \
 fdbrenam.cpp \
 fdict.cpp \
 fdict.h \
 fdllmain.cpp \
 fdynbuf.cpp \
 fdynbuf.h \
 ffilehdr.cpp \
 filesys.h \
 flaimsys.h \
 flbackup.cpp \
 flclose.cpp \
 flconvrt.cpp \
 fldbglog.cpp \
 flerror.cpp \
 flgethdr.cpp \
 flindex.cpp \
 flkeyret.cpp \
 flmstat.cpp \
 flmstat.h \
 flog.cpp \
 flog.h \
 flreduce.cpp \
 fltrabrt.cpp \
 fltrbeg.cpp \
 fltrcmit.cpp \
 f_nici.cpp \
 f_nici.h \
 fnumber.cpp \
 frestore.cpp \
 frow.cpp \
 fsblk_u.cpp \
 fscursor.cpp \
 fscursor.h \
 fsdatacu.cpp \
 fslfile.cpp \
 fslfileu.cpp \
 fsrefupd.cpp \
 fstructs.h \
 fsysdata.cpp \
 funicode.cpp \
 fvector.cpp \
 insertrow.cpp \
 kybldkey.cpp \
 kybuild.cpp \
 kycollat.cpp \
 kyeword.cpp \
 kyqsort.cpp \
 kyunlock.cpp \
 opendatabase.cpp \
 recover.cpp \
 rfl.cpp \
 rfl.h \
 scache.cpp \
 select.cpp \
 sqleval.cpp \
 sqloptimize.cpp \
 sqlquery.h \
 sqlstatement.cpp \
 sqlstatement.h \
 translog.cpp \
 updaterow.cpp \
 whereclause.cpp

libflaimsql_la_CPPFLAGS = $(FTK_INCLUDE)
libflaimsql_la_LIBADD = $(FTK_LTLIB)
libflaimsql_la_LDFLAGS = -version-info 1:0:0
