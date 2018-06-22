# FLM_PROG_TRY_JAR(["quiet"])
# --------------------------
# FLM_PROG_TRY_JAR tests for an existing Java ARchive program.i
# It sets or uses the environment variable JAR.
#
# If no arguments are given to this macro, and no Java jar
# program can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed. Any other
# argument is considered by autoconf to be an error at expansion
# time.
#
# Makes the JAR variable precious to Autoconf. You can 
# use the JAR variable in your Makefile.in files with 
# @JAR@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-08-30
# License:  AllPermissive
#
AC_DEFUN([FLM_PROG_TRY_JAR],
[AC_ARG_VAR([JAR], [Java archive utility])dnl
AC_CHECK_PROGS([JAR], [fastjar jar])
ifelse([$1],,
[if test -z "$JAR"; then
  AC_MSG_WARN([Java ARchive program not found - continuing without jar])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])# FLM_PROG_TRY_JAR
