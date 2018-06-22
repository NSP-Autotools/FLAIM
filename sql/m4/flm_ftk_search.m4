# FLM_FTK_SEARCH
# --------------
# Define AC_ARG_VAR (user variables), FTKLIB and FTKINC, 
# allowing the user to specify the location of the flaim toolkit 
# library and header file. If not specified, check for these files:
#
#   1. As a sub-project.
#   2. As a super-project (sibling to the current project).
#   3. As installed components on the system.
#
# If found, AC_SUBST FTK_LTLIB and FTK_INCLUDE variables with 
# values derived from FTKLIB and FTKINC user variables.
# FTKLIB and FTKINC are file locations, whereas FTK_LTLIB and 
# FTK_INCLUDE are linker and preprocessor command-line options.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-08-30
# License:  AllPermissive
#
AC_DEFUN([FLM_FTK_SEARCH],
[AC_ARG_VAR([FTKLIB], [The PATH wherein libflaimtk.la can be found])
AC_ARG_VAR([FTKINC], [The PATH wherein flaimtk.h can be found])
dnl
# Ensure that both or neither FTK paths were specified.
if { test -n "$FTKLIB" && test -z "$FTKINC"; } || \
   { test -n "$FTKINC" && test -z "$FTKLIB"; } then
  AC_MSG_ERROR([Specify both FTKINC and FTKLIB, or neither.])
fi 

# Not specified? Check for FTK in standard places.
if test -z "$FTKLIB"; then
  # Check for flaim tool kit as a sub-project.
  if test -d "$srcdir/ftk"; then
    AC_CONFIG_SUBDIRS([ftk])
    FTKINC='$(top_srcdir)/ftk/src'
    FTKLIB='$(top_builddir)/ftk/src'
  else
    # Check for flaim tool kit as a super-project.
    if test -d "$srcdir/../ftk"; then
      FTKINC='$(top_srcdir)/../ftk/src'
      FTKLIB='$(top_builddir)/../ftk/src'
    fi
  fi
fi

# Still empty? Check for *installed* flaim tool kit.
if test -z "$FTKLIB"; then
  AC_CHECK_LIB([flaimtk], [ftkFastChecksum], 
    [AC_CHECK_HEADERS([flaimtk.h])
     LIBS="-lflaimtk $LIBS"],
    [AC_MSG_ERROR([No FLAIM Toolkit found. Terminating.])])
fi

# AC_SUBST command line variables from FTKLIB and FTKINC.
if test -n "$FTKLIB"; then
  AC_SUBST([FTK_LTLIB], ["$FTKLIB/libflaimtk.la"])
  AC_SUBST([FTK_INCLUDE], ["-I$FTKINC"])
fi[]dnl
])# FLM_FTK_SEARCH
