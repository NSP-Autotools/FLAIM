#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([FLAIMTK],[1.2],[flaim-users@lists.sourceforge.net])
AM_INIT_AUTOMAKE([-Wall -Werror])
AM_PROG_AR
LT_PREREQ([2.4])
LT_INIT([dlopen])

AC_LANG([C++])

AC_CONFIG_MACRO_DIRS([m4])
AC_CONFIG_SRCDIR([src/flaimtk.h])
AC_CONFIG_HEADERS([config.h])

# Checks for programs.
AC_PROG_CXX
AC_PROG_INSTALL

# Checks for optional programs.
FLM_PROG_TRY_DOXYGEN

# Configure options: --enable-debug[=no].
AC_ARG_ENABLE([debug],
  [AS_HELP_STRING([--enable-debug],
    [enable debug code (default is no)])],
  [debug="$withval"], [debug=no])

# Configure option: --enable-openssl[=no].
AC_ARG_ENABLE([openssl], 
  [AS_HELP_STRING([--enable-openssl], 
    [enable the use of openssl (default is no)])], 
  [openssl="$withval"], [openssl=no])

# Create Automake conditional based on the DOXYGEN variable
AM_CONDITIONAL([HAVE_DOXYGEN], [test -n "$DOXYGEN"])
#AM_COND_IF([HAVE_DOXYGEN], [AC_CONFIG_FILES([docs/doxyfile])])
AS_IF([test -n "$DOXYGEN"], [AC_CONFIG_FILES([docs/doxyfile])])

# Configure for large files, even in 32-bit environments
AC_SYS_LARGEFILE

# Check for pthreads
AX_PTHREAD(
  [AC_DEFINE([HAVE_PTHREAD], [1], [Define if you have POSIX threads libraries and header files.])
   LIBS="$PTHREAD_LIBS $LIBS"
   CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
   CXXFLAGS="$CXXFLAGS $PTHREAD_CXXFLAGS"])

# Checks for libraries.
AC_SEARCH_LIBS([initscr], [ncurses])
AC_CHECK_HEADER([curses.h],,[echo "*** Error: curses.h not found - install curses devel package."; exit 1])
AC_CHECK_LIB([rt], [aio_suspend])
AS_IF([test "x$openssl" = xyes],
  [AC_DEFINE([FLM_OPENSSL], [1], [Define to use openssl])
   AC_CHECK_LIB([ssl], [SSL_new])
   AC_CHECK_LIB([crypto], [CRYPTO_add])
   AC_CHECK_LIB([dl], [dlopen])
   AC_CHECK_LIB([z], [gzopen])])

# Checks for header files.
AC_HEADER_RESOLV
AC_CHECK_HEADERS([arpa/inet.h fcntl.h limits.h malloc.h netdb.h netinet/in.h stddef.h stdlib.h string.h strings.h sys/mount.h sys/param.h sys/socket.h sys/statfs.h sys/statvfs.h sys/time.h sys/vfs.h unistd.h utime.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT32_T
AC_TYPE_MODE_T
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_CHECK_MEMBERS([struct stat.st_blksize])
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT8_T

# Checks for library functions.
AC_FUNC_LSTAT_FOLLOWS_SLASHED_SYMLINK
AC_FUNC_MALLOC
AC_FUNC_MKTIME
AC_CHECK_FUNCS([atexit fdatasync ftruncate getcwd gethostbyaddr gethostbyname gethostname gethrtime gettimeofday inet_ntoa localtime_r memmove memset mkdir pstat_getdynamic realpath rmdir select socket strchr strrchr strstr])

# Configure DEBUG source code, if requested.
AS_IF([test "x$debug" = xyes],
  [AC_DEFINE([FLM_DEBUG], [1], [Define to enable FLAIM debug features])])

# Configure compiler and platform flags and options.
# This stuff was in the original makefile...
# Personally, I believe -O2 does as good a job as 
# a hand-coded set of optimization flags...
gcc_opt_flags="-O -foptimize-sibling-calls
 -fstrength-reduce -fcse-follow-jumps -fcse-skip-blocks
 -frerun-cse-after-loop -frerun-loop-opt -fgcse
 -fgcse-lm -fgcse-sm -fdelete-null-pointer-checks
 -fexpensive-optimizations -fregmove -fsched-interblock
 -fsched-spec -fcaller-saves -fpeephole2 -freorder-blocks
 -freorder-functions -falign-functions -falign-jumps
 -falign-loops -falign-labels -fcrossjumping"

# Enable g++ specific compiler flags.
# Currently, enabling all warnings displays the type-pun error dozens
# of times, and of course enabling warnings as werrors breaks the build
#if test "x$CXX" = "xg++"; then
#  CXXFLAGS="$CXXFLAGS -Wall -Werror"
#  CXXFLAGS="$CXXFLAGS $gcc_opt_flags"
#fi

# Configure supported platforms' compiler and linker flags.
case $host in
  sparc-*-solaris*)
    LDFLAGS="$LDFLAGS -R /usr/lib/lwp"
    case $CXX in
      *g++*) ;;
      *)
        if "x$debug" = xno; then
          CXXFLAGS="$CXXFLAGS -xO3"
        fi
        SUN_STUDIO_COMPILER=`$CXX -V | grep "Sun C++"`
        if "x$SUN_STUDIO_COMPILER" = "xSun C++"; then
          CXXFLAGS="$CXXFLAGS -errwarn=%all -errtags -erroff=hidef,inllargeuse,doubunder"
        fi ;;
    esac ;;

  *-apple-darwin*)
    AC_DEFINE([OSX], [1], [Define if building on Apple OSX.]) ;;

  *-*-aix*)
    case $CXX in
      *g++*) ;;
      *) CXXFLAGS="$CXXFLAGS -qstrict" ;;
    esac ;;

  *-*-hpux*)
    case $CXX in
      *g++*) ;;
      *)
        # Disable "Placement operator delete invocation is not yet implemented" warning
        CXXFLAGS="$CXXFLAGS +W930" ;;
    esac ;;
esac

AC_CONFIG_FILES([Makefile
                 docs/Makefile
                 obs/Makefile
                 obs/flaimtk.spec
                 src/Makefile
                 util/Makefile
                 src/libflaimtk.pc])

AC_OUTPUT

# Fix broken libtool
sed 's/link_all_deplibs=no/link_all_deplibs=yes/' libtool >libtool.tmp && \
  mv libtool.tmp libtool

cat <<EOF

  FLAIM Tool Kit ($PACKAGE_NAME) version $PACKAGE_VERSION
  Prefix.........: $prefix
  Debug Build....: $debug
  Using OpenSSL..: $openssl
  C++ Compiler...: $CXX $CXXFLAGS $CPPFLAGS
  Linker.........: $LD $LDFLAGS $LIBS
  Doxygen........: ${DOXYGEN:-NONE}

EOF
