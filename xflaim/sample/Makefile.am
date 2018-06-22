noinst_PROGRAMS = sample

XFLAIM_INC=-I$(top_srcdir)/src
XFLAIM_LIB=../src/libxflaim.la

sample_SOURCES = sample.cpp
sample_CPPFLAGS = $(XFLAIM_INC) $(FTK_INCLUDE)
sample_LDADD = $(XFLAIM_LIB) $(FTK_LTLIB)
