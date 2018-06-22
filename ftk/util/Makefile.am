FTK_INCLUDE = -I$(top_srcdir)/src
FTK_LTLIB = ../src/libflaimtk.la

check_PROGRAMS = ftktest

ftktest_SOURCES = ftktest.cpp
ftktest_CPPFLAGS = $(FTK_INCLUDE)
ftktest_LDADD = $(FTK_LTLIB)

TESTS = ftktest
