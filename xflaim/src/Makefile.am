if HAVE_JAVA
  JAVADIR = java
  JNI_LIBADD = java/libxfjni.la
endif

if HAVE_CSHARP
  CSDIR = cs
  CSI_LIBADD = cs/libxfcsi.la
endif

SUBDIRS = $(JAVADIR) $(CSDIR)

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libxflaim.pc

lib_LTLIBRARIES = libxflaim.la
include_HEADERS = xflaim.h

libxflaim_la_SOURCES = \
 btreeinfo.cpp \
 f_btpool.cpp \
 f_btpool.h \
 f_btree.cpp \
 f_btree.h \
 fbtrset.cpp \
 fbtrset.h \
 fcache.h \
 f_ccs.cpp \
 f_ccs.h \
 fcollate.cpp \
 fcollate.h \
 fdbcnfig.cpp \
 fdbcopy.cpp \
 fdbremov.cpp \
 fdbrenam.cpp \
 fdict.cpp \
 fdict.h \
 fdllmain.cpp \
 fdoclist.cpp \
 fdom.cpp \
 ffilehdr.cpp \
 filesys.h \
 flaimsys.h \
 flbackup.cpp \
 flblddb.cpp \
 flchkdb.cpp \
 flchkix.cpp \
 flchktr.cpp \
 flclose.cpp \
 flconvrt.cpp \
 flcreate.cpp \
 fldbglog.cpp \
 flerror.cpp \
 flindex.cpp \
 flkeyret.cpp \
 flmstat.cpp \
 flmstat.h \
 flog.cpp \
 flog.h \
 flopen.cpp \
 flreduce.cpp \
 flsweep.cpp \
 fltrabrt.cpp \
 fltrbeg.cpp \
 fltrcmit.cpp \
 flverify.cpp \
 f_nici.cpp \
 fntable.cpp \
 fnumber.cpp \
 fqeval.cpp \
 fqsort.cpp \
 fquery.cpp \
 fquery.h \
 frestore.cpp \
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
 fxml.cpp \
 fxml.h \
 fxpath.cpp \
 fxpath.h \
 kybldkey.cpp \
 kybuild.cpp \
 kycollat.cpp \
 kyeword.cpp \
 kyqsort.cpp \
 kyunlock.cpp \
 ncache.cpp \
 nodeinfo.cpp \
 recover.cpp \
 rfl.cpp \
 rfl.h \
 scache.cpp \
 translog.cpp

libxflaim_la_CPPFLAGS = $(FTK_INCLUDE)
libxflaim_la_LIBADD = $(JNI_LIBADD) $(CSI_LIBADD) $(FTK_LTLIB)
libxflaim_la_LDFLAGS = -version-info 3:2:0
