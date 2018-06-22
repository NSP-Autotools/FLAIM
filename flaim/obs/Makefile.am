rpmspec = $(PACKAGE_TARNAME).spec

rpmmacros =\
 --define="_rpmdir $${PWD}"\
 --define="_srcrpmdir $${PWD}"\
 --define="_sourcedir $${PWD}/.."\
 --define="_specdir $${PWD}"\
 --define="_builddir $${PWD}"

RPMBUILD = rpmbuild
RPMFLAGS = --nodeps --buildroot="$${PWD}/_rpm"

rpmcheck:
	if [ which rpmbuild &> /dev/null ]; then \
	  echo "*** This make target requires an rpm-based linux distribution."; \
	  (exit 1); exit 1; \
	fi

srcrpm: rpmcheck $(rpmspec)
	$(RPMBUILD) $(RPMFLAGS) -bs $(rpmmacros) $(rpmspec)

rpms: rpmcheck $(rpmspec)
	$(RPMBUILD) $(RPMFLAGS) -ba $(rpmmacros) $(rpmspec)

.PHONY: rpmcheck srcrpm rpms

