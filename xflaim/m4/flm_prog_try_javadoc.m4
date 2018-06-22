# FLM_PROG_TRY_JAVADOC(["quiet"])
# ------------------------------
# FLM_PROG_TRY_JAVADOC tests for an existing javadoc generator.
# It uses or sets the environment variable JAVADOC.
#
# If no arguments are given to this macro, and no javadoc 
# program can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed. Any other
# argument is considered by autoconf to be an error at expansion
# time.
#
# Makes the JAVADOC variable precious to Autoconf. You can 
# use the JAVADOC variable in your Makefile.in files with 
# @JAVADOC@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-08-30
# License:  AllPermissive
#
AC_DEFUN([FLM_PROG_TRY_JAVADOC],
[AC_ARG_VAR([JAVADOC], [Java source documentation utility])dnl
AC_CHECK_PROGS([JAVADOC], [gjdoc javadoc])
ifelse([$1],,
[if test -z "$JAVADOC"; then
  AC_MSG_WARN([Javadoc program not found - continuing without javadoc])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])# FLM_PROG_TRY_JAVADOC
