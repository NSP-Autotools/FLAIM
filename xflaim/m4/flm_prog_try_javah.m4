# FLM_PROG_TRY_JAVAH(["quiet"])
# ----------------------------
# FLM_PROG_TRY_JAVAH tests for an existing Java native header (JNI)
# generator. It uses or sets the environment variable JAVAH.
#
# If no arguments are given to this macro, and no javah
# program can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed. Any other
# argument is considered by autoconf to be an error at expansion
# time.
#
# Makes the JAVAH variable precious to Autoconf. You can 
# use the JAVAH variable in your Makefile.in files with 
# @JAVAH@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-08-30
# License:  AllPermissive
#
AC_DEFUN([FLM_PROG_TRY_JAVAH],
[AC_ARG_VAR([JAVAH], [Java header utility])dnl
AC_CHECK_PROGS([JAVAH], [gcjh javah])
ifelse([$1],,
[if test -z "$JAVAH"; then
  AC_MSG_WARN([Java header program not found - continuing without javah])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])# FLM_PROG_TRY_JAVAH
