ACLOCAL_AMFLAGS = -I m4

EXTRA_DIST = README.W32 tools win32

SUBDIRS = ftk flaim sql xflaim

rpms srcrpm:
	for dir in $(SUBDIRS); do \
 	  (cd $$dir && $(MAKE) $(AM_MAKEFLAGS) $@) || exit 1; \
	done

.PHONY: rpms srcrpm
