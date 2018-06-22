# FLM_PROG_TRY_CSC(["quiet"])
# --------------------------
# FLM_PROG_TRY_CSC tests for an existing CSharp compiler. It sets
# or uses the environment variable CSC.
#
# It checks for a Mono CSharp compiler (msc) and then for a 
# Microsoft CSharp compiler (csc).
#
# If no arguments are given to this macro, and no CSharp
# compiler can be found, it prints a warning message to STDOUT
# and to the config.log file. If the "quiet" argument is passed, 
# then only the normal "check" line is displayed. Any other
# argument is considered by autoconf to be an error at expansion
# time.
#
# Makes the CSC variable precious to Autoconf. You can 
# use the CSC variable in your Makefile.in files with 
# @CSC@.
#
# Author:   John Calcote <john.calcote@gmail.com>
# Modified: 2009-08-30
# License:  AllPermissive
#
AC_DEFUN([FLM_PROG_TRY_CSC],
[AC_ARG_VAR([CSC], [CSharp compiler])dnl
AC_CHECK_PROGS([CSC], [mcs csc])
ifelse([$1],,
[if test -z "$CSC"; then
  AC_MSG_WARN([CSharp compiler not found - continuing without CSharp])
fi], [$1], [quiet],, [m4_fatal([Invalid option '$1' in $0])])
])# FLM_PROG_TRY_CSC
