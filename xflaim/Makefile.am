ACLOCAL_AMFLAGS = -I m4

EXTRA_DIST = GNUMakefile README.W32 debian netware win32

SUBDIRS = src util sample docs obs

doc_DATA = AUTHORS ChangeLog COPYING INSTALL NEWS README

RPM = rpm

rpms srcrpm: dist
	(cd obs && $(MAKE) $(AM_MAKEFLAGS) $@) || exit 1
	rpmarch=`$(RPM) --showrc | grep "^build arch" | sed 's/\(.*: \)\(.*\)/\2/'`; \
	test -z "obs/$$rpmarch" || ( mv obs/$$rpmarch/* . && rm -rf obs/$$rpmarch )
	rm -rf obs/$(distdir)

.PHONY: srcrpm rpms
