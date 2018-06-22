# FLM_PROG_TRY_JNI(["quiet"])
# --------------------------
# FLM_PROG_TRY_JNI tests for the existence of the three
# tools required to build Java Native Interface (JNI) 
# modules: javac, javah, and jar. It manages the
# environment variable flm_prog_have_jni.
#
# If all three tools are present, then the 
# flm_prog_have_jni environment variable is set to 'yes', 
# otherwise it's set to 'no'.
#
# If no arguments are given to this macro, and any of
# these programs cannot be found, it prints a warning 
# message to STDOUT and to the config.log file. If the 
# "quiet" argument is passed, then only the normal 
# "check" line is displayed. Any other argument is 
# considered by autoconf to be an error at expansion
# time.
#
# Makes the JAVAC, JAVAH, and JAR variables precious to
# Autoconf. You can use these variables in your Makefile.in
# files with @JAVAC@, @JAVAH@, and @JAR@, respectively.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-08-30
# License:  AllPermissive
#
AC_DEFUN([FLM_PROG_TRY_JNI],
[FLM_PROG_TRY_JAVAC([quiet])dnl
FLM_PROG_TRY_JAVAH([quiet])dnl
FLM_PROG_TRY_JAR([quiet])dnl
ifelse([$1],,
[flm_prog_have_jni=yes
if test -z "$JAVAC"; then flm_prog_have_jni=no; fi
if test -z "$JAVAH"; then flm_prog_have_jni=no; fi
if test -z "$JAR"; then flm_prog_have_jni=no; fi
if test "x$flm_prog_have_jni" = xno; then
  AC_MSG_WARN([Some required JNI tools are missing - continuing without JNI support])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])# FLM_PROG_TRY_JNI
