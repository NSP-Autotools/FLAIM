#------------------------------------------------------------------------------
# Desc:	GNU makefile for XFLAIM library and utilities
# Tabs:	3
#
# Copyright (c) 2000-2007 Novell, Inc. All Rights Reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; version 2.1
# of the License.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, contact Novell, Inc.
#
# To contact Novell about this file by physical or electronic mail, 
# you may find current contact information at www.novell.com.
#
# $Id$
#------------------------------------------------------------------------------

#############################################################################
#
# Sample Usage:
#
#		make clean debug all
#
#############################################################################

# -- Include --

-include config.in

# -- Project --

project_name = xflaim
project_display_name = XFLAIM
project_brief_desc = Embeddable XML Database Engine

# -- Maintainers --

ahodgkinson_info = Andrew Hodgkinson
dsanders_info = Daniel Sanders

# -- Versions --

major_version = 5
minor_version = 1

version = $(major_version).$(minor_version).$(svn_revision)

# libtool versions are updated according to the following rules:
#
# 1. Start with a version of 0.0.0 for a library
# 2. Update the version number immediately before a public release of the software
# 3. If the library source code has changed at all since the last update, increment revision
# 4. If any interfaces have been added, removed, or changed since the last update, increment current
# 5. If any interfaces have been added since the last public release, increment age
# 6. If any interfaces have been removed since the last public release, set age to 0

so_current = 3
so_revision = 2
so_age = 0
shared_lib_version =

package_release_num = 1

# -- Paths initializations --

install_prefix = /usr

# -- RPM, SPEC file names

package_proj_name = lib$(project_name)
package_proj_name_and_ver = $(package_proj_name)-$(version)

# -- Determine if we are only cleaning --

util_targets = checkdb rebuild view sample xshell
test_targets = basictest binarytest colldeftest dictchangetest dictdeftest \
		dirtyexittest1 dirtyexittest2 domnodetest enctest importtest \
		indexdeftest indextest1 indextest2 indextest3 \
		namespacetest metaphonetest regressiontest rfltest sortkeytest \
		sortkeytest2 xpathtest1 xpathtest2
all_targets = java csharp rpms install libs all allutils test $(util_targets) $(test_targets)
found_targets = $(foreach target,$(MAKECMDGOALS),$(if $(findstring $(target),$(all_targets)),$(target),))

ifneq (,$(findstring clean,$(MAKECMDGOALS)))
	do_clean = 1
	ifeq ($(if $(findstring 0,$(words $(found_targets))),1,0),0)
      $(error Cannot specify other targets with clean target)
	endif
else
	do_clean = 0
endif

# -- Target variables --

target_build_type =
usenativecc = yes
target_os_family =
target_processor_family =
target_word_size =
requested_word_size =
win_target =
unix_target =
netware_target =
submake_targets =
netware_ring_0_target =
sparc_generic =
debian_arch = unknown
with_openssl =

# -- Enable command echoing --

ifneq (,$(findstring verbose,$(MAKECMDGOALS)))
	submake_targets += verbose
	ec =
else
	ec = @
endif

# -- Determine the host operating system --

ifndef host_os_family
	ifneq (,$(findstring WIN,$(OS)))
		host_os_family = win
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring Win,$(OS)))
		host_os_family = win
	endif
endif

ifndef host_os_family
	ifeq (,$(OSTYPE))
		ifneq (,$(RPM_OS))
			OSTYPE = $(RPM_OS)
		endif
	endif
	
	ifeq (,$(OSTYPE))
		OSTYPE := $(shell uname -s)
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring Linux,$(OSTYPE)))
		host_os_family = linux
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring linux,$(OSTYPE)))
		host_os_family = linux
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring solaris,$(OSTYPE)))
		host_os_family = solaris
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring SunOS,$(OSTYPE)))
		host_os_family = solaris
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring darwin,$(OSTYPE)))
		host_os_family = osx
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring Darwin,$(OSTYPE)))
		host_os_family = osx
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring aix,$(OSTYPE)))
		host_os_family = aix
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring hpux,$(OSTYPE)))
		host_os_family = hpux
	endif
endif

ifndef host_os_family
	ifneq (,$(findstring HP-UX,$(OSTYPE)))
		host_os_family = hpux
	endif
endif

ifndef host_os_family
   $(error Host operating system could not be determined.  You may need to export OSTYPE from the environment.)
endif

# -- Target build type --

ifndef target_build_type
	ifneq (,$(findstring debug,$(MAKECMDGOALS)))
		submake_targets += debug
		target_build_type = debug
	endif
endif

ifndef target_build_type
	ifneq (,$(findstring release,$(MAKECMDGOALS)))
		submake_targets += release
		target_build_type = release
	endif
endif

ifndef target_build_type
	target_build_type = release
endif

# -- Use non-native (i.e., gcc) compiler on Solaris, etc.

ifneq (,$(findstring usegcc,$(MAKECMDGOALS)))
	submake_targets += usegcc
	usenativecc = no
endif

# -- Override platform default word size? --

ifneq (,$(findstring 64bit,$(MAKECMDGOALS)))
	submake_targets += 64bit
	requested_word_size = 64
endif

ifneq (,$(findstring 32bit,$(MAKECMDGOALS)))
	submake_targets += 32bit
	requested_word_size = 32
endif

# -- Target operating system --

ifndef target_os_family
	ifeq ($(host_os_family),linux)
		unix_target = yes
		target_os_family = linux
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),solaris)
		unix_target = yes
		target_os_family = solaris
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),osx)
		unix_target = yes
		target_os_family = osx
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),aix)
		unix_target = yes
		target_os_family = aix
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),hpux)
		unix_target = yes
		target_os_family = hpux
	endif
endif

ifneq (,$(findstring nlm,$(MAKECMDGOALS)))
	submake_targets += nlm
	netware_target = yes
	target_os_family = netware
	host_os_family = win
	
	ifneq (,$(findstring ring0,$(MAKECMDGOALS)))
		submake_targets += ring0
		netware_ring_0_target = yes
	endif
endif

ifndef target_os_family
	ifeq ($(host_os_family),win)
		win_target = yes
		target_os_family = win
	endif
endif

ifndef target_os_family
   $(error Target operating system could not be determined)
endif

# -- Host word size and processor --

host_native_word_size =
host_processor_family =
host_supported_word_sizes =

ifneq (,$(PROCESSOR_ARCHITECTURE))
	HOSTTYPE = $(PROCESSOR_ARCHITECTURE)
endif

ifeq (,$(HOSTTYPE))
	ifneq (,$(RPM_ARCH))
		HOSTTYPE = $(RPM_ARCH)
	endif
endif

ifeq (,$(HOSTTYPE))
	ifneq ($(host_os_family),hpux)
		ifneq ($(host_os_family),linux)
			HOSTTYPE := $(shell uname -p)
		endif
	endif
	ifeq (,$(HOSTTYPE))
		HOSTTYPE := $(shell uname -m)
	else
		ifneq (,$(findstring nvalid,$(HOSTTYPE)))
			HOSTTYPE := $(shell uname -m)
		else
			ifneq (,$(findstring unknown,$(HOSTTYPE)))
				HOSTTYPE := $(shell uname -m)
			endif
		endif
	endif
endif

ifeq (,$(HOSTTYPE))
	$(error HOSTTYPE environment variable has not been set)
endif

ifndef host_native_word_size
	ifneq (,$(findstring x86_64,$(HOSTTYPE)))
		host_processor_family = x86
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring amd64,$(HOSTTYPE)))
		host_processor_family = x86
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring x86,$(HOSTTYPE)))
		host_processor_family = x86
		host_native_word_size = 32
		host_supported_word_sizes = 32
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring 86,$(HOSTTYPE)))
		host_processor_family = x86
		host_native_word_size = 32
		host_supported_word_sizes = 32
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring ia64,$(HOSTTYPE)))
		host_processor_family = ia64
		host_native_word_size = 64
		host_supported_word_sizes = 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring s390x,$(HOSTTYPE)))
		host_processor_family = s390
		host_native_word_size = 64
		host_supported_word_sizes = 31 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring s390,$(HOSTTYPE)))
		host_processor_family = s390
		host_native_word_size = 31
		host_supported_word_sizes = 31
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring ppc64,$(HOSTTYPE)))
		host_processor_family = powerpc
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring ppc,$(HOSTTYPE)))
		host_processor_family = powerpc
		host_native_word_size = 32
		host_supported_word_sizes = 32
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring sparc,$(HOSTTYPE)))
		host_processor_family = sparc
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring powerpc,$(HOSTTYPE)))
		host_processor_family = powerpc
		host_native_word_size = 32
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring Power,$(HOSTTYPE)))
		host_processor_family = powerpc
		host_native_word_size = 32
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring rs6000,$(HOSTTYPE)))
		host_processor_family = powerpc
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring hppa,$(HOSTTYPE)))
		host_processor_family = hppa
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
	ifneq (,$(findstring 9000,$(HOSTTYPE)))
		host_processor_family = hppa
		host_native_word_size = 64
		host_supported_word_sizes = 32 64
	endif
endif

ifndef host_native_word_size
   $(error Unable to determine host word size. $(HOSTTYPE))
endif

# -- Target word size and processor --

ifneq (,$(findstring nlm,$(MAKECMDGOALS)))
	target_processor_family = x86
	target_word_size = 32
	target_supported_word_sizes = 32
else
	target_processor_family = $(host_processor_family)
	target_word_size = $(host_native_word_size)
	target_supported_word_sizes = $(host_supported_word_sizes)
endif

ifdef requested_word_size
	ifneq (,$(findstring $(requested_word_size),$(target_supported_word_sizes)))
		target_word_size = $(requested_word_size)
	else
      $(error Unsupported target word size)
	endif
endif

# -- Debian architecture --

ifeq ($(target_os_family),linux)
	ifeq ($(target_processor_family),x86)
		ifeq ($(target_word_size),64)
			debian_arch = amd64
		else
			debian_arch = i386
		endif
	endif
	ifeq ($(target_processor_family),sparc)
		debian_arch = sparc
	endif
	ifeq ($(target_processor_family),powerpc)
		debian_arch = powerpc
	endif
endif

# -- Other targets and options --

ifneq (,$(findstring sparcgeneric,$(MAKECMDGOALS)))
	sparc_generic = yes
endif

ifneq (,$(findstring openssl,$(MAKECMDGOALS)))
	submake_targets += openssl
	with_openssl = yes
endif

# -- Helper functions --

define normpath
$(strip $(subst \,/,$(1)))
endef

ifeq (win,$(host_os_family))
define hostpath
$(strip $(subst /,\,$(1)))
endef
else
define hostpath
$(strip $(1))
endef
endif

ifeq (win,$(host_os_family))
define ppath
$(strip $(subst \,\\,$(subst /,\,$(1))))
endef
else
define ppath
$(strip $(1))
endef
endif

ifeq (win,$(host_os_family))
   define create_archive
		-$(ec)$(call rmcmd,$(2))
		$(ec)cmd /C "cd $(call hostpath,$(1)) && $(call hostpath,$(tooldir)/7z) a -ttar -r $(call hostpath,$(2)).tar $(call hostpath,$(3))"
		$(ec)cmd /C "cd $(call hostpath,$(1)) && $(call hostpath,$(tooldir)/7z) a -tgzip -r $(call hostpath,$(2)).tar.gz $(call hostpath,$(2)).tar"
		$(ec)cmd /C "cd $(call hostpath,$(1)) && del $(call hostpath,$(2)).tar"
   endef

   define extract_archive
		$(ec)cmd /C "cd $(call hostpath,$(1)) && $(call hostpath,$(tooldir)/7z) x -y $(call hostpath,$(2)).tar.gz
		$(ec)cmd /C "cd $(call hostpath,$(1)) && $(call hostpath,$(tooldir)/7z) x -y $(call hostpath,$(2)).tar
   endef
else
   define create_archive
		-$(ec)$(call rmcmd,$(2))
		$(ec)tar cf $(2).tar -C $(1) $(3)
		$(ec)gzip -f $(2).tar
		$(ec)chmod 775 $(2).tar.gz
   endef

   define extract_archive
		$(ec)gunzip -f $(strip $(1))/$(2).tar.gz
		$(ec)tar xvf $(strip $(1))/$(2).tar -C $(1)
   endef
endif

# Platform-specific commands, directories, etc.

ifeq ($(host_os_family),win)
	allprereqs  = $(call hostpath,$+)
	copycmd = copy /Y $(call hostpath,$(1)) $(call hostpath,$(2)) 1>NUL
	dircopycmd = xcopy /Y /E /V /I $(call hostpath,$(1)) $(call hostpath,$(2))
	rmcmd = if exist $(call hostpath,$(1)) del /Q $(call hostpath,$(1)) 1>NUL
	rmdircmd = if exist $(call hostpath,$(1)) rmdir /q /s $(call hostpath,$(1)) 1>NUL
	mkdircmd = -if not exist $(call hostpath,$(1)) mkdir $(call hostpath,$(1))
	runtest = cmd /C "cd $(call hostpath,$(test_dir)) && $(1) -d"
	topdir := $(call normpath,$(shell chdir))
else
	allprereqs = $+
	copycmd = cp -f $(1) $(2)
	dircopycmd = cp -rf $(1) $(2)
	rmcmd = rm -f $(1)
	rmdircmd = rm -rf $(1)
	mkdircmd = mkdir -p $(1)
	runtest = sh -c "cd $(test_dir); ./$(1) -d; exit"
	topdir := $(shell pwd)
endif

# If this is an un-tar'd or un-zipped source package, the tools directory
# will be subordinate to the top directory.  Otherwise, it will be
# a sibling to the top directory - which is how it is set up in the
# subversion repository.

ifeq "$(wildcard $(topdir)/tools*)" ""
	tooldir := $(dir $(topdir))tools/$(host_os_family)
else
	tooldir := $(topdir)/tools/$(host_os_family)
endif

# -- Utility variables --

em :=
sp := $(em) $(em)
percent := \045
dollar := \044
question := \077
asterisk := \052
dash := \055
backslash := \134
double_quote := \042

# -- printf --

ifdef unix_target
	gprintf = printf
else
	gprintf = $(call hostpath,$(tooldir)/printf.exe)
endif

# Determine the toolkit directory

ifeq "$(wildcard $(topdir)/ftk)" ""
	ftk_dir := $(dir $(topdir))ftk
else
	ftk_dir := $(topdir)/ftk
endif

ftk_src_dir = $(ftk_dir)/src

# -- Subversion Revision --

calc_svn_revision =
ignore_local_mods =

ifneq (,$(findstring ignore-local-mods,$(MAKECMDGOALS)))
	submake_targets += ignore-local-mods
	ignore_local_mods = 1
endif

ifneq (,$(findstring ilm,$(MAKECMDGOALS)))
	submake_targets += ilm
	ignore_local_mods = 1
endif

ifdef ignore_local_mods
	local_mods_ok = 1
else
	local_mods_ok =
endif

ifneq (,$(findstring dist,$(MAKECMDGOALS)))
	calc_svn_revision = 1
	ifndef ignore_local_mods
		local_mods_ok =
	endif
endif

ifneq (,$(findstring rpm,$(MAKECMDGOALS)))
	calc_svn_revision = 1
	ifndef ignore_local_mods
		local_mods_ok =
	endif
endif

ifneq (,$(findstring ubuntusrc,$(MAKECMDGOALS)))
	calc_svn_revision = 1
	ifndef ignore_local_mods
		local_mods_ok =
	endif
endif

ifneq (,$(findstring docs,$(MAKECMDGOALS)))
	calc_svn_revision = 1
	ifndef ignore_local_mods
		local_mods_ok =
	endif
endif

ifneq (,$(findstring changelog,$(MAKECMDGOALS)))

	calc_svn_revision = 1

	# Get the info for this directory

	ifndef svn_user
      $(error Must define svn_user=<user> in environment or as a parameter)
	endif

	ifndef svn_rev
      $(error Must define svn_rev=<low[:high]> in environment or as a parameter)
	endif

	svnrevs = $(subst :, ,$(svn_rev))
	svn_low_rev = $(word 1,$(svnrevs))
	svn_high_rev = $(word 2,$(svnrevs))

	svnurl0 := $(shell svn info)
	svnurl1 = $(subst URL: ,URL:,$(svnurl0))
	svnurl2 = $(filter URL:%,$(svnurl1))
	svnurl3 = $(subst URL:,,$(svnurl2))
	svnurl = $(subst ://,://$(svn_user)@,$(svnurl3))
endif

ifdef calc_svn_revision

	# Get the info for all files.

	ifndef local_mods_ok
		srevision := $(shell svnversion . -n)

		ifneq (,$(findstring M,$(srevision)))
         $(error Local modifications found - please check in before making distro)
		endif

		ifneq (,$(findstring :,$(srevision)))
         $(error Mixed revisions in repository - please update before making distro)
		endif

		srevision := $(shell svnversion $(ftk_dir) -n)

		ifneq (,$(findstring M,$(srevision)))
         $(error Local modifications found - please check in before making distro)
		endif

		ifneq (,$(findstring :,$(srevision)))
         $(error Mixed revisions in repository - please update before making distro)
		endif
	endif

	numdigits = $(words $(subst 9,9 ,$(subst 8,8 ,$(subst 7,7 ,\
						$(subst 6,6 ,$(subst 5,5 ,$(subst 4,4 ,$(subst 3,3 ,\
						$(subst 2,2 ,$(subst 1,1 ,$(subst 0,0 ,$(1))))))))))))
	revision0 := $(shell svn info -R . $(ftk_dir))
	revision1 = $(subst Last Changed Rev: ,LastChangedRev:,$(revision0))
	revision2 = $(filter LastChangedRev:%,$(revision1))
	revision3 = $(subst LastChangedRev:,,$(revision2))
	revision4 = $(sort $(revision3))
	revision5 = $(foreach num,$(revision4),$(call numdigits,$(num)):$(num))
	revision6 = $(sort $(revision5))
	revision7 = $(word $(words $(revision6)),$(revision6))
	svn_revision = $(word 2,$(subst :, ,$(revision7)))

else
	ifeq "$(wildcard SVNRevision.*)" ""
		svn_revision = 0
	else
		svn_revision = $(word 2,$(subst ., ,$(wildcard SVNRevision.*)))
	endif
endif

ifeq "$(svn_high_rev)" ""
	svn_high_rev = $(svn_revision)
endif

# Files and Directories

ifeq ($(target_word_size),64)
	ifeq ($(target_os_family),linux)
		lib_dir_name = lib64
	endif
endif

ifndef lib_dir_name
	lib_dir_name = lib
endif

ifndef rpm_build_root
	ifneq (,$(DESTDIR))
		rpm_build_root = $(DESTDIR)
	else
		rpm_build_root =
	endif
endif

lib_install_dir = $(rpm_build_root)$(install_prefix)/$(lib_dir_name)
include_install_dir = $(rpm_build_root)$(install_prefix)/include
pkgconfig_install_dir = $(lib_install_dir)/pkgconfig
build_output_dir = $(topdir)/build
doxygen_output_dir = $(build_output_dir)/docs
javadoc_output_dir = $(build_output_dir)/javadocs

target_path = $(build_output_dir)/$(target_os_family)-$(target_processor_family)-$(target_word_size)/$(target_build_type)

package_dir = $(target_path)/package
spec_dir = $(package_dir)/SPECS
spec_file = $(spec_dir)/$(package_proj_name).spec
package_sources_dir = $(package_dir)/SOURCES
package_bin_dir = $(package_dir)/BIN
package_build_dir = $(package_dir)/BUILD
package_rpms_dir = $(package_dir)/RPMS
package_srpms_dir = $(package_dir)/SRPMS
pkgconfig_file_name = $(package_proj_name).pc
pkgconfig_file = $(package_dir)/$(pkgconfig_file_name)

package_debian_dir = $(package_dir)/DEBIAN
debian_stage_dir = $(package_dir)/debian_stage
debian_pkginfo_dir = $(debian_stage_dir)/DEBIAN

package_ubuntu_dir = $(package_dir)/UBUNTU
package_version_ubuntu = $(version)-0ubuntu1
package_distro_ubuntu = edgy
ubuntu_stage_dir = $(package_dir)/ubuntu_stage
ubuntu_pkginfo_dir = $(ubuntu_stage_dir)/DEBIAN

package_stage_parent_dir = $(package_dir)/stage
package_stage_dir = $(package_stage_parent_dir)/$(package_proj_name_and_ver)
package_bin_stage_dir = $(package_stage_parent_dir)/$(package_proj_name_and_ver)/$(target_os_family)-$(target_processor_family)-$(target_word_size)/$(target_build_type)
package_lib_stage_dir = $(package_bin_stage_dir)/lib
package_shared_lib_stage_dir = $(package_lib_stage_dir)/shared
package_static_lib_stage_dir = $(package_lib_stage_dir)/static
package_util_stage_dir = $(package_bin_stage_dir)/util
package_inc_stage_dir = $(package_stage_parent_dir)/$(package_proj_name_and_ver)/include

src_package_dir = $(package_sources_dir)
bin_package_dir = $(package_bin_dir)

src_package_base_name = $(package_proj_name_and_ver)
bin_package_base_name = $(package_proj_name_and_ver)-$(target_os_family)-$(target_processor_family)-$(target_word_size)-bin

src_package_name=$(src_package_base_name).tar.gz
bin_package_name=$(bin_package_base_name).tar.gz

rpm_name = $(package_proj_name_and_ver)-$(package_release_num).$(HOSTTYPE).rpm
srpm_name = $(package_proj_name_and_ver)-$(package_release_num).src.rpm
develrpm_name = $(package_proj_name)-devel-$(version)-$(package_release_num).$(HOSTTYPE).rpm

java_output_dir = $(target_path)/java
java_class_output_dir = $(target_path)/java/classes
csharp_output_dir = $(target_path)/csharp
cstest_output_dir = $(target_path)/csharp/cstest

ifdef JAVA_HOME
	jhome := $(subst \,/,$(JAVA_HOME))
endif

inc_dirs = src util $(ftk_src_dir)

ifdef jhome
	jni_src_dir = java/jni
	inc_dirs += $(jni_src_dir)
endif

xflaim_jni_jar = $(java_output_dir)/xflaimjni.jar
xflaim_jar_manifest = $(java_output_dir)/xflaimjni.mf

xflaim_csharp_assembly = $(csharp_output_dir)/xflaim_csharp.dll
xflaim_csharp_assembly_xml = $(csharp_output_dir)/xflaim_csharp.xml
cstest_exe = $(cstest_output_dir)/cstest.exe

util_dir = $(target_path)/util
test_dir = $(target_path)/test
sample_dir = $(target_path)/sample
csharp_dir = $(target_path)/csharp
lib_dir = $(target_path)/$(lib_dir_name)
shared_lib_dir = $(lib_dir)/shared
static_lib_dir = $(lib_dir)/static

util_obj_dir = $(util_dir)/obj
test_obj_dir = $(test_dir)/obj
sample_obj_dir = $(sample_dir)/obj

lib_obj_dir = $(lib_dir)/obj
csharp_obj_dir = $(csharp_dir)/obj

doxyfile = $(doxygen_output_dir)/Doxyfile

# -- Tools --

libr =
exe_linker =
shared_linker =
compiler =

# Compiler definitions and flags

ccflags =
clrflags =
noclrflags =
cs_flags = /nologo /warn:4 /warnaserror+
cs_defines =
ifeq ($(target_build_type),debug)
	cs_flags += /debug+ /debug:full /define:FLM_DEBUG
else
	cs_flags += /optimize+
endif

ccdefs =

ifeq ($(target_word_size),64)
	ccdefs += FLM_64BIT
endif

##############################################################################
#   Win settings
##############################################################################
ifdef win_target
	exe_suffix = .exe
	obj_suffix = .obj
	lib_prefix =
	static_lib_suffix = .lib
	shared_lib_suffix = .dll
	libr = lib.exe
	exe_linker = link.exe
	shared_linker = link.exe
	compiler = cl.exe
	csharp_compiler := csc.exe

	# Compiler defines and flags

	ccflags += /nologo /c /GF /GR /J /MD /W4 /WX /Zi
	ccdefs += _CRT_SECURE_NO_DEPRECATE
	ccdefs += WIN32_LEAN_AND_MEAN
	ccdefs += WIN32_EXTRA_LEAN

	ifeq ($(target_build_type),debug)
		ccflags += /Ob1 /Od /RTC1 /Wp64
		noclrflags += /RTC1
		ccdefs += FLM_DEBUG
	else
		ccflags += /O2
	endif

	# Linker switches

	shared_link_flags = \
		/DLL \
		/DEBUG /PDB:$(call hostpath,$(@:.dll=.pdb)) \
		/map:$(call hostpath,$(@:.dll=.map)) \
		/INCREMENTAL:NO \
		/NOLOGO \
		/OUT:$(call hostpath,$@)

	exe_link_flags = \
		/DEBUG /PDB:$(call hostpath,$(@:.exe=.pdb)) \
		/map:$(call hostpath,$(@:.exe=.map)) \
		/INCREMENTAL:NO \
		/FIXED:NO \
		/NOLOGO \
		/OUT:$(call hostpath,$@)
		
	# Java

	ifdef jhome	
		inc_dirs += $(subst $(sp),@,"$(jhome)/include")
		inc_dirs += $(subst $(sp),@,"$(jhome)/include/win32")
	endif

	# Libraries that our various components need to link against

	link_libs = imagehlp.lib user32.lib rpcrt4.lib wsock32.lib advapi32.lib

	# Convert the list of defines into a proper set of command-line params

	ifdef ccdefs
		ccdefine = $(foreach def,$(strip $(ccdefs)),/D$(def))
	endif

	# Same thing for the include dirs

	ccinclude = $(foreach inc_dir,$(strip $(inc_dirs)),/I$(subst @,$(sp),$(call hostpath,$(inc_dir))))

	# Concatenate everything into the ccflags variable

	ccflags += $(ccdefine) $(ccinclude)
endif

##############################################################################
#   Linux/Unix settings
##############################################################################
ifdef unix_target

	ifneq ($(so_age),0)
		shared_lib_version = $(so_current).$(so_revision).$(so_age)
	else
		ifneq ($(so_revision),0)
			shared_lib_version = $(so_current).$(so_revision)
		else
			shared_lib_version = $(so_current)
		endif
	endif
	
	exe_suffix =
	obj_suffix = .o
	lib_prefix = lib
	static_lib_suffix = .a
	shared_lib_suffix = .so.$(shared_lib_version)
	
	compiler = g++
	exe_linker = g++
	shared_linker = g++

	ifeq ($(target_os_family),osx)
		libr = libtool
	else
		libr = ar
	endif

	gcc_optimization_flags = \
		-O \
		-foptimize-sibling-calls \
		-fstrength-reduce -fcse-follow-jumps \
		-fcse-skip-blocks \
		-frerun-cse-after-loop \
		-frerun-loop-opt \
		-fgcse \
		-fgcse-lm \
		-fgcse-sm \
		-fdelete-null-pointer-checks \
		-fexpensive-optimizations \
		-fregmove \
		-fsched-interblock \
		-fsched-spec \
		-fcaller-saves \
		-fpeephole2 \
		-freorder-blocks \
		-freorder-functions \
		-falign-functions \
		-falign-jumps \
		-falign-loops \
		-falign-labels \
		-fcrossjumping
				
	ifeq ($(usenativecc),yes)
		ifeq ($(target_os_family),solaris)
			compiler = CC
			exe_linker = CC
			shared_linker = CC
			compiler_version := $(shell $(compiler) -V 2>&1)	
			ifneq (,$(findstring Sun C++,$(compiler_version)))
				sun_studio_compiler = yes
			endif
		endif
	endif

	ifeq ($(usenativecc),yes)
		ifeq ($(target_os_family),aix)
			compiler = xlC_r
			exe_linker = xlC_r
			shared_linker = xlC_r
		endif
	endif

	ifeq ($(usenativecc),yes)
		ifeq ($(target_os_family),hpux)
			compiler = aCC
			exe_linker = aCC
			shared_linker = aCC
		endif
	endif

	# Compiler defines and flags

	ifeq ($(compiler),g++)
		ccflags += -Wall -Werror -fPIC
		ifneq ($(target_processor_family),ia64)
			ccflags += -m$(target_word_size)
		endif
	endif

	ifeq ($(target_os_family),linux)
		csharp_compiler := mcs
		cs_defines = /d:mono

		# Must support 64 bit file sizes - even for 32 bit builds.

		ccdefs += N_PLAT_UNIX _LARGEFILE64_SOURCE _FILE_OFFSET_BITS=64

		ifdef jhome	
			inc_dirs += $(subst $(sp),@,"$(jhome)/include")
			inc_dirs += $(subst $(sp),@,"$(jhome)/include/linux")
		endif
		
		ifeq ($(target_build_type),release)
			ccflags += $(gcc_optimization_flags)
		endif
	endif

	ifeq ($(target_os_family),solaris)
		ifdef jhome	
			inc_dirs += $(subst $(sp),@,"$(jhome)/include")
			inc_dirs += $(subst $(sp),@,"$(jhome)/include/solaris")
		endif

		ifeq ($(usenativecc),yes)
			ccflags += -KPIC
			ifeq ($(target_build_type),release)
				ccflags += -xO3
			endif
			ifeq ($(sun_studio_compiler),yes)
				ccflags += -errwarn=%all -errtags -erroff=hidef,inllargeuse,doubunder
			endif
			ifeq ($(target_word_size),64)
				ccflags += -xarch=generic64
			else
				# Must support 64 bit file sizes - even for 32 bit builds.

				ccdefs += _LARGEFILE_SOURCE _FILE_OFFSET_BITS=64
				
				ifdef sparc_generic
					ccflags += -xarch=generic
					ccdefs += FLM_SPARC_GENERIC
				else
					ccflags += -xarch=v8plus
				endif
			endif
		endif
	endif

	ifeq ($(target_os_family),aix)
		ifeq ($(usenativecc),yes)
			ccflags += -qthreaded -qstrict
			ifeq ($(target_word_size),64)
				ccflags += -q64
			else
				# Must support 64 bit file sizes - even for 32 bit builds.

				ccflags += -q32
				ccdefs += _LARGEFILE_SOURCE _FILE_OFFSET_BITS=64
			endif
		endif
	endif

	ifeq ($(target_os_family),hpux)
		ifeq ($(usenativecc),yes)
		
				# Disable "Placement operator delete invocation is not yet
				# implemented" warning
		
				ccflags += +W930
		
			ifeq ($(target_word_size),64)
				ccflags += +DD64
			else
				# Must support 64 bit file sizes - even for 32 bit builds.
	
				ccdefs += _LARGEFILE_SOURCE _FILE_OFFSET_BITS=64
			endif
		endif
	endif

	ifeq ($(target_os_family),osx)
		ccdefs += OSX
		
		ifeq ($(target_build_type),release)
			ccflags += $(gcc_optimization_flags)
		endif
		inc_dirs += /System/Library/Frameworks/JavaVM.framework/Versions/CurrentJDK/Headers
	endif

	ccdefs += _REENTRANT

	ifeq ($(target_build_type),debug)
		ccdefs += FLM_DEBUG
		ccflags += -g
	endif

	# Use OpenSSL
	
	ifdef with_openssl
		ccdefs += FLM_OPENSSL
	endif

	# Convert the list of defines into a proper set of command-line params

	ifdef ccdefs
		ccdefine = $(foreach def,$(strip $(ccdefs)),-D$(def))
	endif

	# Same thing for the include dirs

	ccinclude = $(foreach inc_dir,$(strip $(inc_dirs)),-I$(subst @,$(sp),$(inc_dir)))

	# Concatenate everything into the ccflags variable

	ccflags += $(ccdefine) $(ccinclude)

	# Linker switches

	shared_link_flags =
	link_flags = -o $@
	libr_flags =

	ifeq ($(compiler),g++)
		ifneq ($(target_processor_family),ia64)
			shared_link_flags += -m$(target_word_size)
			link_flags += -m$(target_word_size)
		endif
	endif

	link_libs = -lpthread

	ifeq ($(target_os_family),linux)
		link_libs += -lrt -lstdc++ -ldl -lncurses
		ifdef with_openssl
			link_libs +=  -lssl -lcrypto -ldl -lz
		endif
		
		shared_link_flags += -shared -Wl,-Bsymbolic -fpic \
			-Wl,-soname,$(@F) -o $@
	endif

	ifeq ($(target_os_family),solaris)
		link_flags += -R /usr/lib/lwp
		shared_link_flags += -G -pic -o $@
		
		ifeq ($(usenativecc),yes)
			ifeq ($(target_word_size),64)
				link_flags += -xarch=generic64
				shared_link_flags += -xarch=generic64
			else
				link_flags += -xarch=v8plus
			endif
		endif

		link_libs += -lm -lc -ldl -lsocket -lnsl -lrt -lcurses
	endif

	ifeq ($(target_os_family),aix)
		ifeq ($(target_word_size),64)
			link_flags += -q64
			libr_flags = -X64
		else
			link_flags += -q32
			libr_flags = -X32
		endif
		
		link_libs += -lm -lc -lcurses
	endif

	ifeq ($(target_os_family),hpux)
		ifeq ($(target_word_size),64)
			link_flags += +DD64
		endif
		link_libs += -lm -lc -lrt -lcurses
	endif

	ifeq ($(target_os_family),osx)
		shared_lib_suffix = -$(major_version).$(so_current).dylib
		link_libs += -lstdc++ -ldl -lncurses
		shared_link_flags += -dynamiclib
		shared_link_flags += -current_version $(major_version).$(so_current).$(so_revision)
		shared_link_flags += -compatibility_version $(major_version).$(so_current).0
		shared_link_flags += -o $@
	endif

	exe_link_flags = $(link_flags)
endif

##############################################################################
#   NetWare settings
##############################################################################
ifdef netware_target
	exe_suffix = .nlm
	obj_suffix = .obj
	lib_prefix =
	static_lib_suffix = .lib
	shared_lib_suffix = .nlm

	ifdef WATCOM
		wc_dir = $(WATCOM)
	endif

	ifdef watcom
		wc_dir = $(watcom)
	endif

	ifndef wc_dir
		wc_dir = $(WC_DIR)
	endif

	ifndef wc_dir
      $(error Watcom compiler could not be found.  Please define wc_dir)
	endif
	
	wc_dir := $(call normpath,$(wc_dir))
	
	ifndef netware_ring_0_target
		ifndef ndk_dir
         $(error Netware SDK could not be found. Please define ndk_dir)
		endif
		
		ndk_dir := $(call normpath,$(ndk_dir))
	endif

	libr = "$(call normpath,$(strip $(wc_dir)))/binnt/wlib.exe"
	exe_linker = "$(call normpath,$(strip $(wc_dir)))/binnt/wlink.exe"
	shared_linker = "$(call normpath,$(strip $(wc_dir)))/binnt/wlink.exe"
	compiler = "$(call normpath,$(wc_dir))/binnt/wpp386.exe"

	ifneq ($(target_build_type),release)
		ccdefs += FLM_DEBUG
	endif
	
	ifdef netware_ring_0_target
		ccdefs += FLM_RING_ZERO_NLM
	else
		ccdefs += FLM_LIBC_NLM
	endif

	ccflags += /ez /6s /w4 /za /zq /zm /s /ei /of+ /we /bt=NETWARE

	ifeq ($(target_build_type),release)
		ccflags += /oair
	else
		ccflags += /hc
	endif

	libflags += /b /q /p=256
	link_flags = /m /l /v /s
	
	inc_dirs += $(ndk_dir)/libc/include \
					$(ndk_dir)/libc/include/winsock

	export include = $(foreach inc_dir,$(strip $(inc_dirs)),$(call hostpath,$(inc_dir));)
	export INCLUDE = $(include)
	export wpp386 = /d$(subst $(sp), /d,$(strip $(ccdefs))) $(ccflags)
	export wcc386 = /d$(subst $(sp), /d,$(strip $(ccdefs))) $(ccflags)

   define make_ring_0_lis_file_cmd
		$(ec)$(gprintf) "option verbose\n" > $(4)
		$(ec)$(gprintf) "option stack=32k\n" >> $(4)
		$(ec)$(gprintf) "option nod\n" >> $(4)
		$(ec)$(gprintf) "option map\n" >> $(4)
		$(ec)$(gprintf) "option nodefaultlibs\n" >> $(4)
		$(ec)$(gprintf) "option screenname 'NONE'\n" >> $(4)
		$(ec)$(gprintf) "option threadname '$(2)'\n" >> $(4)
		$(ec)$(gprintf) "option start = f_nlmEntryPoint\n" >> $(4)
		$(ec)$(gprintf) "option exit = f_nlmExitPoint\n" >> $(4)
		$(ec)$(gprintf) "option nodefaultlibs\n" >> $(4)
		$(ec)$(gprintf) "option xdcdata=nlm.xdc\n" >> $(4)
		$(ec)$(gprintf) "option pseudopreemption\n" >> $(4) 
		$(ec)$(gprintf) "debug all debug novell\n" >> $(4)
		$(ec)$(gprintf) "form novell nlm '$(2)'\n" >> $(4)
		$(ec)$(gprintf) "name $(call ppath,$(1)/$(2)$(exe_suffix))\n" >> $(4)
		$(ec)$(gprintf) "file $(subst $(sp),\nfile ,$(call ppath,$(3)))\n" >> $(4)
		$(ec)$(gprintf) "library $(call ppath,$(xflaim_static_lib))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(ctorarst)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(dtorarst)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(undefed)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(undefmbd)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(pure_err)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(stablcl)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(stabact)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(stabactv)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(stabmod)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(prwdata)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(moddtorr)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/plib3s.lib)(stabadt)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/netware/clib3s.lib)(i8d)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/netware/clib3s.lib)(i8m)\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(wc_dir)/lib386/netware/clib3s.lib)(i8s)\n" >> $(4)
		$(ec)$(gprintf) "alias __wcpp_4_fatal_runtime_error_=f_fatalRuntimeError\n" >> $(4)
   endef
	
   define make_libc_lis_file_cmd
		$(ec)$(gprintf) "option verbose\n" > $(4)
		$(ec)$(gprintf) "option stack=32k\n" >> $(4)
		$(ec)$(gprintf) "option nod\n" >> $(4)
		$(ec)$(gprintf) "option map\n" >> $(4)
		$(ec)$(gprintf) "option nodefaultlibs\n" >> $(4)
		$(ec)$(gprintf) "option screenname 'NONE'\n" >> $(4)
		$(ec)$(gprintf) "option threadname '$(2)'\n" >> $(4)
		$(ec)$(gprintf) "option start = _LibCPrelude\n" >> $(4)
		$(ec)$(gprintf) "option exit = _LibCPostlude\n" >> $(4)
		$(ec)$(gprintf) "option nodefaultlibs\n" >> $(4)
		$(ec)$(gprintf) "option xdcdata=nlm.xdc\n" >> $(4)
		$(ec)$(gprintf) "option pseudopreemption\n" >> $(4) 
		$(ec)$(gprintf) "debug all debug novell\n" >> $(4)
		$(ec)$(gprintf) "form novell nlm '$(2)'\n" >> $(4)
		$(ec)$(gprintf) "name $(call ppath,$(1)/$(2)$(exe_suffix))\n" >> $(4)
		$(ec)$(gprintf) "file $(subst $(sp),\nfile ,$(call ppath,$(3)))\n" >> $(4)
		$(ec)$(gprintf) "file $(call ppath,$(ndk_dir)/libc/imports/libcpre.obj)\n" >> $(4)
		$(ec)$(gprintf) "library $(call ppath,$(xflaim_static_lib))\n" >> $(4)
		$(ec)$(gprintf) "library $(call ppath,$(wc_dir)/lib386/plib3s.lib)\n" >> $(4)
		$(ec)$(gprintf) "library $(call ppath,$(wc_dir)/lib386/netware/libc3s.lib)\n" >> $(4)
   endef
	
   define make_ring_0_imp_file_cmd
		$(ec)$(gprintf) "import __WSAFDIsSet\n" > $(1)
		$(ec)$(gprintf) "import ActivateScreen\n" >> $(1)
		$(ec)$(gprintf) "import Alloc\n" >> $(1)
		$(ec)$(gprintf) "import AllocateResourceTag\n" >> $(1)
		$(ec)$(gprintf) "import atomic_dec\n" >> $(1)
		$(ec)$(gprintf) "import atomic_inc\n" >> $(1)
		$(ec)$(gprintf) "import atomic_xchg\n" >> $(1)
		$(ec)$(gprintf) "import BitTest\n" >> $(1)
		$(ec)$(gprintf) "import CEvaluateExpression\n" >> $(1)
		$(ec)$(gprintf) "import CFindLoadModuleHandle\n" >> $(1)
		$(ec)$(gprintf) "import CheckKeyStatus\n" >> $(1)
		$(ec)$(gprintf) "import ClearScreen\n" >> $(1)
		$(ec)$(gprintf) "import CloseFile\n" >> $(1)
		$(ec)$(gprintf) "import CloseScreen\n" >> $(1)
		$(ec)$(gprintf) "import CMovB\n" >> $(1)
		$(ec)$(gprintf) "import CMoveFast\n" >> $(1)
		$(ec)$(gprintf) "import ConvertPathString\n" >> $(1)
		$(ec)$(gprintf) "import ConvertSecondsToTicks\n" >> $(1)
		$(ec)$(gprintf) "import ConvertTicksToSeconds\n" >> $(1)
		$(ec)$(gprintf) "import CpuCurrentProcessor\n" >> $(1)
		$(ec)$(gprintf) "import CreateDirectory\n" >> $(1)
		$(ec)$(gprintf) "import CreateFile\n" >> $(1)
		$(ec)$(gprintf) "import CSetD\n" >> $(1)
		$(ec)$(gprintf) "import DebuggerSymbolList\n" >> $(1)
		$(ec)$(gprintf) "import DeleteDirectory\n" >> $(1)
		$(ec)$(gprintf) "import DirectorySearch\n" >> $(1)
		$(ec)$(gprintf) "import DirectReadFile\n" >> $(1)
		$(ec)$(gprintf) "import DirectReadFile\n" >> $(1)
		$(ec)$(gprintf) "import DirectWriteFile\n" >> $(1)
		$(ec)$(gprintf) "import DirectWriteFile\n" >> $(1)
		$(ec)$(gprintf) "import DirectWriteFileNoWait\n" >> $(1)
		$(ec)$(gprintf) "import DirectWriteFileNoWait\n" >> $(1)
		$(ec)$(gprintf) "import DisableInputCursor\n" >> $(1)
		$(ec)$(gprintf) "import DisplayScreenTextWithAttribute\n" >> $(1)
		$(ec)$(gprintf) "import DOSFirstByteBitMap\n" >> $(1)
		$(ec)$(gprintf) "import EnableInputCursor\n" >> $(1)
		$(ec)$(gprintf) "import EnterDebugger\n" >> $(1)
		$(ec)$(gprintf) "import EraseFile\n" >> $(1)
		$(ec)$(gprintf) "import ExpandFileInContiguousBlocks\n" >> $(1)
		$(ec)$(gprintf) "import ExpandFileInContiguousBlocks\n" >> $(1)
		$(ec)$(gprintf) "import ExportPublicSymbol\n" >> $(1)
		$(ec)$(gprintf) "import FindAndLoadNLM\n" >> $(1)
		$(ec)$(gprintf) "import Free\n" >> $(1)
		$(ec)$(gprintf) "import FreeLimboVolumeSpace\n" >> $(1)
		$(ec)$(gprintf) "import FreeLimboVolumeSpace\n" >> $(1)
		$(ec)$(gprintf) "import GetCacheBufferSize\n" >> $(1)
		$(ec)$(gprintf) "import GetClosestSymbol\n" >> $(1)
		$(ec)$(gprintf) "import GetCurrentClock\n" >> $(1)
		$(ec)$(gprintf) "import GetCurrentNumberOfCacheBuffers\n" >> $(1)
		$(ec)$(gprintf) "import GetCurrentTime\n" >> $(1)
		$(ec)$(gprintf) "import GetEntryFromPathStringBase\n" >> $(1)
		$(ec)$(gprintf) "import GetEntryFromPathStringBase\n" >> $(1)
		$(ec)$(gprintf) "import GetFileSize\n" >> $(1)
		$(ec)$(gprintf) "import GetKey\n" >> $(1)
		$(ec)$(gprintf) "import GetNLMAllocMemoryCounts\n" >> $(1)
		$(ec)$(gprintf) "import GetOriginalNumberOfCacheBuffers\n" >> $(1)
		$(ec)$(gprintf) "import GetProductMajorVersionNumber\n" >> $(1)
		$(ec)$(gprintf) "import GetRunningProcess\n" >> $(1)
		$(ec)$(gprintf) "import GetScreenSize\n" >> $(1)
		$(ec)$(gprintf) "import GetSyncClockFields\n" >> $(1)
		$(ec)$(gprintf) "import GetSystemConsoleScreen\n" >> $(1)
		$(ec)$(gprintf) "import ImportPublicSymbol\n" >> $(1)
		$(ec)$(gprintf) "import kCreateThread\n" >> $(1)
		$(ec)$(gprintf) "import kCurrentThread\n" >> $(1)
		$(ec)$(gprintf) "import kDelayThread\n" >> $(1)
		$(ec)$(gprintf) "import kDestroyThread\n" >> $(1)
		$(ec)$(gprintf) "import kExitThread\n" >> $(1)
		$(ec)$(gprintf) "import kGetThreadName\n" >> $(1)
		$(ec)$(gprintf) "import kGetThreadName\n" >> $(1)
		$(ec)$(gprintf) "import KillMe\n" >> $(1)
		$(ec)$(gprintf) "import kMutexAlloc\n" >> $(1)
		$(ec)$(gprintf) "import kMutexFree\n" >> $(1)
		$(ec)$(gprintf) "import kMutexLock\n" >> $(1)
		$(ec)$(gprintf) "import kMutexUnlock\n" >> $(1)
		$(ec)$(gprintf) "import kReturnCurrentProcessorID\n" >> $(1)
		$(ec)$(gprintf) "import kScheduleThread\n" >> $(1)
		$(ec)$(gprintf) "import kSemaphoreAlloc\n" >> $(1)
		$(ec)$(gprintf) "import kSemaphoreExamineCount\n" >> $(1)
		$(ec)$(gprintf) "import kSemaphoreFree\n" >> $(1)
		$(ec)$(gprintf) "import kSemaphoreSignal\n" >> $(1)
		$(ec)$(gprintf) "import kSemaphoreTimedWait\n" >> $(1)
		$(ec)$(gprintf) "import kSemaphoreWait\n" >> $(1)
		$(ec)$(gprintf) "import kSetThreadLoadHandle\n" >> $(1)
		$(ec)$(gprintf) "import kSetThreadName\n" >> $(1)
		$(ec)$(gprintf) "import kYieldIfTimeSliceUp\n" >> $(1)
		$(ec)$(gprintf) "import kYieldThread\n" >> $(1)
		$(ec)$(gprintf) "import LoadModule\n" >> $(1)
		$(ec)$(gprintf) "import LoadRules\n" >> $(1)
		$(ec)$(gprintf) "import MapFileHandleToFCB\n" >> $(1)
		$(ec)$(gprintf) "import MapPathToDirectoryNumber\n" >> $(1)
		$(ec)$(gprintf) "import MapPathToDirectoryNumber\n" >> $(1)
		$(ec)$(gprintf) "import MapVolumeNameToNumber\n" >> $(1)
		$(ec)$(gprintf) "import ModifyDirectoryEntry\n" >> $(1)
		$(ec)$(gprintf) "import ModifyDirectoryEntry\n" >> $(1)
		$(ec)$(gprintf) "import MountVolume\n" >> $(1)
		$(ec)$(gprintf) "import NDSCreateStreamFile\n" >> $(1)
		$(ec)$(gprintf) "import NDSDeleteStreamFile\n" >> $(1)
		$(ec)$(gprintf) "import NDSOpenStreamFile\n" >> $(1)
		$(ec)$(gprintf) "import NWLocalToUnicode\n" >> $(1)
		$(ec)$(gprintf) "import NWUnicodeToLocal\n" >> $(1)
		$(ec)$(gprintf) "import OpenFile\n" >> $(1)
		$(ec)$(gprintf) "import OpenScreen\n" >> $(1)
		$(ec)$(gprintf) "import OutputToScreen\n" >> $(1)
		$(ec)$(gprintf) "import PositionInputCursor\n" >> $(1)
		$(ec)$(gprintf) "import PositionOutputCursor\n" >> $(1)
		$(ec)$(gprintf) "import ReadFile\n" >> $(1)
		$(ec)$(gprintf) "import RenameEntry\n" >> $(1)
		$(ec)$(gprintf) "import RestartServer\n" >> $(1)
		$(ec)$(gprintf) "import ReturnResourceTag\n" >> $(1)
		$(ec)$(gprintf) "import ReturnVolumeMappingInformation\n" >> $(1)
		$(ec)$(gprintf) "import ReturnVolumeMappingInformation\n" >> $(1)
		$(ec)$(gprintf) "import RevokeFileHandleRights\n" >> $(1)
		$(ec)$(gprintf) "import SetCursorStyle\n" >> $(1)
		$(ec)$(gprintf) "import SetFileSize\n" >> $(1)
		$(ec)$(gprintf) "import SetFileSize\n" >> $(1)
		$(ec)$(gprintf) "import SGUIDCreate\n" >> $(1)
		$(ec)$(gprintf) "import SizeOfAllocBlock\n" >> $(1)
		$(ec)$(gprintf) "import SwitchToDirectFileMode\n" >> $(1)
		$(ec)$(gprintf) "import SwitchToDirectFileMode\n" >> $(1)
		$(ec)$(gprintf) "import UngetKey\n" >> $(1)
		$(ec)$(gprintf) "import UnImportPublicSymbol\n" >> $(1)
		$(ec)$(gprintf) "import UnloadRules\n" >> $(1)
		$(ec)$(gprintf) "import VMGetDirectoryEntry\n" >> $(1)
		$(ec)$(gprintf) "import WriteFile\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_bind\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_closesocket\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_gethostbyaddr\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_gethostbyname\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_gethostname\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_htonl\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_htons\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_inet_addr\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_inet_ntoa\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_listen\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_recv\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_select\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_send\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_setsockopt\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_shutdown\n" >> $(1)
		$(ec)$(gprintf) "import WS2_32_socket\n" >> $(1)
		$(ec)$(gprintf) "import WSAAccept\n" >> $(1)
		$(ec)$(gprintf) "import WSACleanup\n" >> $(1)
		$(ec)$(gprintf) "import WSAConnect\n" >> $(1)
		$(ec)$(gprintf) "import WSAGetLastError\n" >> $(1)
		$(ec)$(gprintf) "import WSAStartup\n" >> $(1)
   endef
	
   define make_libc_imp_file_cmd
		$(ec)$(gprintf) "import CurrentProcess\n" > $(1)
		$(ec)$(gprintf) "import @$(call ppath,$(ndk_dir)/libc/imports/libc.imp)\n" >> $(1)
		$(ec)$(gprintf) "import @$(call ppath,$(ndk_dir)/libc/imports/netware.imp)\n" >> $(1)
		$(ec)$(gprintf) "import @$(call ppath,$(ndk_dir)/libc/imports/ws2nlm.imp)\n" >> $(1)
   endef
	
   define flm_exe_link_cmd
		$(call $(if $(netware_ring_0_target),make_ring_0_imp_file_cmd,make_libc_imp_file_cmd),$(call hostpath,$(1)/$(2).imp))
		$(call $(if $(netware_ring_0_target),make_ring_0_lis_file_cmd,make_libc_lis_file_cmd),$(1),$(2),$(3),$(call hostpath,$(1)/$(2).lis))
		$(ec)$(call hostpath,$(exe_linker)) @$(call hostpath,$(1)/$(2).lis) @$(call hostpath,$(1)/$(2).imp)
		$(ec)$(call rmcmd,$(target_path)/$(1).lis)
		$(ec)$(call rmcmd,$(target_path)/$(1).imp)
   endef

endif

# -- File lists --

xflaim_src = \
	$(patsubst src/%.cpp,%.cpp,$(wildcard src/*.cpp))

ftk_src = \
	$(patsubst $(ftk_src_dir)/%.cpp,%.cpp,$(wildcard $(ftk_src_dir)/*.cpp))

xflaim_jni_src = \
	$(patsubst java/jni/%.cpp,%.cpp,$(wildcard java/jni/*.cpp))
	
xflaim_csharp_src = \
	$(patsubst csharp/xflaim/%.cpp,%.cpp,$(wildcard csharp/xflaim/*.cpp))

util_common_src = \
	flm_dlst.cpp \
	flm_lutl.cpp \
	sharutil.cpp

checkdb_src = \
	checkdb.cpp \
	$(util_common_src)

rebuild_src = \
	rebuild.cpp \
	$(util_common_src)

view_src = \
	view.cpp \
	viewblk.cpp \
	viewdisp.cpp \
	viewedit.cpp \
	viewhdr.cpp \
	viewlfil.cpp \
	viewmenu.cpp \
	viewsrch.cpp \
	$(util_common_src)
	
sample_src = \
	sample.cpp \
	$(util_common_src)

xshell_src = \
	fdomedt.cpp \
	xshell.cpp \
	fshell.cpp \
	$(util_common_src)
	
ut_basictest_src = \
	flmunittest.cpp \
	basictestsrv.cpp
	
ut_binarytest_src = \
	flmunittest.cpp \
	binarytest.cpp
	
ut_colldeftest_src = \
	flmunittest.cpp \
	colldeftestsrv.cpp
	
ut_dictchangetest_src = \
	flmunittest.cpp \
	dictchangetest.cpp
	
ut_dictdeftest_src = \
	flmunittest.cpp \
	dictdeftestsrv.cpp
	
ut_dirtyexittest1_src = \
	flmunittest.cpp \
	dirtyexittest1srv.cpp
	
ut_dirtyexittest2_src = \
	flmunittest.cpp \
	dirtyexittest2srv.cpp
	
ut_domnodetest_src = \
	flmunittest.cpp \
	domnodetestsrv.cpp
	
ut_enctest_src = \
	flmunittest.cpp \
	enctestsrv.cpp
	
ut_importtest_src = \
	flmunittest.cpp \
	importtestsrv.cpp
	
ut_indexdeftest_src = \
	flmunittest.cpp \
	indexdeftestsrv.cpp
	
ut_indextest1_src = \
	flmunittest.cpp \
	indextest1srv.cpp
	
ut_indextest2_src = \
	flmunittest.cpp \
	indextest2srv.cpp
	
ut_indextest3_src = \
	flmunittest.cpp \
	indextest3.cpp
	
ut_metaphonetest_src = \
	flmunittest.cpp \
	metaphonetestsrv.cpp
	
ut_namespacetest_src = \
	flmunittest.cpp \
	namespacetestsrv.cpp
	
ut_regressiontest_src = \
	flmunittest.cpp \
	regressiontest.cpp
	
ut_rfltest_src = \
	flmunittest.cpp \
	rfltestsrv.cpp
	
ut_sortkeytest_src = \
	flmunittest.cpp \
	sortkeytest.cpp
	
ut_sortkeytest2_src = \
	flmunittest.cpp \
	sortkeytest2.cpp
	
ut_xpathtest1_src = \
	flmunittest.cpp \
	xpathtest1srv.cpp
	
ut_xpathtest2_src = \
	flmunittest.cpp \
	xpathtest2srv.cpp
	
# -- XFLAIM library --

ftk_obj = $(patsubst %.cpp,$(lib_obj_dir)/%$(obj_suffix),$(ftk_src))
xflaim_obj = $(patsubst %.cpp,$(lib_obj_dir)/%$(obj_suffix),$(xflaim_src))
xflaim_csharp_obj = $(patsubst %.cpp,$(csharp_obj_dir)/%$(obj_suffix),$(xflaim_csharp_src))
xflaim_static_lib = $(static_lib_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
ifndef netware_target
   xflaim_shared_lib = $(shared_lib_dir)/$(lib_prefix)$(project_name)$(shared_lib_suffix)
   xflaim_shared_imp_lib = $(shared_lib_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
endif
ifdef jhome
	xflaim_jni_obj = $(patsubst %.cpp,$(lib_obj_dir)/%$(obj_suffix),$(xflaim_jni_src))
endif

# -- Unit tests  --

ut_basictest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_basictest_src))
ut_binarytest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_binarytest_src))
ut_colldeftest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_colldeftest_src))
ut_dictchangetest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_dictchangetest_src))
ut_dictdeftest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_dictdeftest_src))
ut_dirtyexittest1_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_dirtyexittest1_src))
ut_dirtyexittest2_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_dirtyexittest2_src))
ut_domnodetest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_domnodetest_src))
ut_enctest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_enctest_src))
ut_importtest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_importtest_src))
ut_indexdeftest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_indexdeftest_src))
ut_indextest1_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_indextest1_src))
ut_indextest2_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_indextest2_src))
ut_indextest3_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_indextest3_src))
ut_metaphonetest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_metaphonetest_src))
ut_namespacetest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_namespacetest_src))
ut_regressiontest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_regressiontest_src))
ut_rfltest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_rfltest_src))
ut_sortkeytest_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_sortkeytest_src))
ut_sortkeytest2_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_sortkeytest2_src))
ut_xpathtest1_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_xpathtest1_src))
ut_xpathtest2_obj = $(patsubst %.cpp,$(test_obj_dir)/%$(obj_suffix),$(ut_xpathtest2_src))

# -- Java --

jni_java_src = $(wildcard java/xflaim/*.java)

xedit_java_src = $(wildcard java/util/xedit/*.java)
xedit_java_exe = $(java_class_output_dir)/xedit/XEdit.class

# -- CSharp --

csharp_src = $(wildcard csharp/xflaim/*.cs)
cstest_src = $(wildcard csharp/cstest/*.cs)

# -- Utilities  --

checkdb_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(checkdb_src))
checkdb_exe = $(util_dir)/checkdb$(exe_suffix)

rebuild_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(rebuild_src))
rebuild_exe = $(util_dir)/rebuild$(exe_suffix)

view_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(view_src))
view_exe = $(util_dir)/view$(exe_suffix)

xshell_obj = $(patsubst %.cpp,$(util_obj_dir)/%$(obj_suffix),$(xshell_src))
xshell_exe = $(util_dir)/xshell$(exe_suffix)

sample_obj = $(patsubst %.cpp,$(sample_obj_dir)/%$(obj_suffix),$(sample_src))
sample_exe = $(sample_dir)/sample$(exe_suffix)

# -- Make system pattern search paths -- 

vpath %.cpp src util sample java/jni csharp/xflaim $(ftk_src_dir) 
vpath %.java jni/java/xflaim
vpath %.cs csharp/xflaim csharp/cstest

# -- Default target --

.PHONY : libs
libs: status clean dircheck $(xflaim_static_lib) $(xflaim_shared_lib)

# -- JNI (Java) --

ifdef jhome
java_manifest: dircheck
	$(ec)$(gprintf) "Creating manifest for jar file ...\n"
	$(ec)$(gprintf) "Manifest-version: 1.0\n\n" > $(xflaim_jar_manifest)
	$(ec)$(gprintf) "Name: $(project_name)\n" >> $(xflaim_jar_manifest)
	$(ec)$(gprintf) "Specification-Title: \"XFLAIM Java Classes\"\n" >> $(xflaim_jar_manifest)
	$(ec)$(gprintf) "Specification-Version: \"$(major_version).$(minor_version)\"\n" >> $(xflaim_jar_manifest)
	$(ec)$(gprintf) "Specification-Vendor: \"Novell, Inc.\"\n" >> $(xflaim_jar_manifest)
	$(ec)$(gprintf) "Package-Title: \"xflaim\"\n" >> $(xflaim_jar_manifest)
	$(ec)$(gprintf) "Package-Version: \"$(version)\"\n" >> $(xflaim_jar_manifest)
	$(ec)$(gprintf) "Package-Vendor: \"Novell, Inc.\"\n" >> $(xflaim_jar_manifest)

$(xflaim_jni_jar) : $(jni_java_src) java_manifest
	$(ec)$(gprintf) "Compiling JNI java sources ...\n"
	$(ec)$(call hostpath,$(jhome)/bin/javac) -g -d $(java_class_output_dir) $(call hostpath,$(jni_java_src))
	$(ec)$(gprintf) "Creating JNI jar file ...\n"
	$(ec)$(call rmcmd,$(xflaim_jni_jar))
	$(ec)$(call hostpath,$(jhome)/bin/jar) cfm $(call hostpath,$(xflaim_jni_jar)) \
		$(call hostpath,$(xflaim_jar_manifest)) \
		-C $(call hostpath,$(java_class_output_dir)) $(project_name)
	$(ec)$(gprintf) "Creating JNI header files ...\n"
	$(ec)$(call hostpath,$(jhome)/bin/javah) -d $(jni_src_dir) \
		-classpath $(call hostpath,$(xflaim_jni_jar)) \
		$(foreach jclass,$(notdir $(basename $(jni_java_src))),xflaim.$(jclass))
endif

ifdef jhome
$(xedit_java_exe): $(xedit_java_src) $(xflaim_jni_jar)
	$(ec)$(gprintf) "Compiling XEDIT java sources ...\n"
	$(ec)$(call hostpath,$(jhome)/bin/javac) -g -classpath $(call hostpath,$(xflaim_jni_jar)) \
		-d $(call hostpath,$(java_class_output_dir)) $(call hostpath,$(xedit_java_src))
endif

# -- CSharp --
ifdef csharp_compiler
$(xflaim_csharp_assembly) : $(csharp_src)
	$(ec)$(gprintf) "Compiling CSharp sources ...\n"
	$(ec)$(csharp_compiler) /target:library /out:$(call hostpath,$(xflaim_csharp_assembly)) \
		/doc:$(call hostpath,$(xflaim_csharp_assembly_xml)) $(cs_defines) $(cs_flags)\
		$(call hostpath,$(csharp_src))
endif

ifdef csharp_compiler
$(cstest_exe) : $(cstest_src) $(xflaim_csharp_assembly)
	$(ec)$(gprintf) "Compiling CSTest sources ...\n"
	$(ec)$(csharp_compiler) /out:$(call hostpath,$(cstest_exe)) $(call hostpath,$(cstest_src)) \
		/reference:$(call hostpath,$(xflaim_csharp_assembly)) $(cs_defines) $(cs_flags)
endif

# -- *.cpp -> *$(obj_suffix) --

ifdef win_target
define cpp_to_obj_template
$$($(1)_obj_dir)/%$$(obj_suffix) : %.cpp
	$$(ec)$$(compiler) $$(ccflags) $$(noclrflags) /Fo$$(call hostpath,$$@) $$(call hostpath,$$<)
endef

define cpp_to_clr_obj_template
$$($(1)_obj_dir)/%$$(obj_suffix) : %.cpp
	$$(ec)$$(compiler) $$(ccflags) /clr /AI$$(call hostpath,$$(csharp_output_dir)) /Fo$$(call hostpath,$$@) $$(call hostpath,$$<)
endef
endif

ifdef unix_target
define cpp_to_obj_template
$$($(1)_obj_dir)/%$$(obj_suffix) : %.cpp
	$$(ec)$$(gprintf) "$$<\n"
	$$(ec)$$(compiler) $$(ccflags) -c $$< -o $$@
endef

define cpp_to_clr_obj_template
$$($(1)_obj_dir)/%$$(obj_suffix) : %.cpp
	$$(ec)$$(gprintf) "$$<\n"
	$$(ec)$$(compiler) $$(ccflags) -c $$< -o $$@
endef
endif

ifdef netware_target
define cpp_to_obj_template
$$($(1)_obj_dir)/%$(obj_suffix) : %.cpp
	$$(ec)$$(gprintf) "$$(notdir $$(strip $$@))\n"
	$$(ec)$$(call hostpath,$$(compiler)) $$(call hostpath,$$<) /fo=$$(call hostpath,$$@)
endef

define cpp_to_clr_obj_template
$$($(1)_obj_dir)/%$(obj_suffix) : %.cpp
	$$(ec)$$(gprintf) "$$(notdir $$(strip $$@))\n"
	$$(ec)$$(call hostpath,$$(compiler)) $$(call hostpath,$$<) /fo=$$(call hostpath,$$@)
endef
endif

$(foreach tmpl,lib csharp util sample test,$(eval $(call cpp_to_obj_template,$(tmpl))))
#$(foreach tmpl,csharp,$(eval $(call cpp_to_clr_obj_template,$(tmpl))))

# -- xflaim.lib and libxflaim.a --

$(xflaim_static_lib) : $(xflaim_obj) $(ftk_obj)
	$(ec)$(gprintf) "Building $@ ...\n"
ifdef win_target
	$(ec)$(libr) /NOLOGO $(call hostpath,$+) /OUT:$(call hostpath,$@)
endif
ifdef unix_target
	$(ec)rm -f $@
ifeq ($(target_os_family),osx)
	$(ec)$(libr) -static -o $@ $+
else
	$(ec)$(libr) $(libr_flags) -rcs $@ $+
endif
endif
ifdef netware_target
	$(ec)dir /s/b  $(call hostpath,$(lib_obj_dir)/*$(obj_suffix)) > $(call hostpath,$(static_lib_dir)/xflmlib.lis)
	$(ec)$(call hostpath,$(libr)) $(libflags) $(call hostpath,$(xflaim_static_lib)) @$(call hostpath,$(static_lib_dir)/xflmlib.lis)
endif

# -- xflaim.dll and libxflaim.so --

$(xflaim_shared_lib) : $(xflaim_obj) $(ftk_obj) $(xflaim_jni_obj) $(xflaim_csharp_obj)
	$(ec)$(gprintf) "Building $@ ...\n"
ifdef win_target
	$(ec)$(shared_linker) $(call hostpath,$+) $(shared_link_flags) $(link_libs)
endif
ifdef unix_target
	$(ec)rm -f $@
	$(ec)$(shared_linker) $+ $(shared_link_flags) $(link_libs)
endif

# -- xflaim.lib import library --

$(xflaim_shared_imp_lib) : $(xflaim_shared_lib)
	$(ec)$(gprintf) "Building $@ ...\n"

# -- Utility link command --

ifndef flm_exe_link_cmd
   define flm_exe_link_cmd
		$(ec)$(exe_linker) $(exe_link_flags) $(allprereqs) $(link_libs)
   endef
endif

# -- checkdb --

.PHONY : checkdb
checkdb: status clean dircheck libs $(checkdb_exe)
$(checkdb_exe): $(checkdb_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(util_dir),checkdb,$(checkdb_obj))

# -- rebuild --

.PHONY : rebuild
rebuild: status clean dircheck libs $(rebuild_exe)
$(rebuild_exe): $(rebuild_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(util_dir),rebuild,$(rebuild_obj))

# -- view --

.PHONY : view
view: status clean dircheck libs $(view_exe)
$(view_exe): $(view_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(util_dir),view,$(view_obj))

# -- sample --

.PHONY : sample
sample: status clean dircheck libs $(sample_exe)
$(sample_exe): $(sample_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(sample_dir),sample,$(sample_obj))
	$(ec)$(call copycmd,sample/xmlfiles/*.xml,$(sample_dir))

# -- xshell --

.PHONY : xshell
xshell: status clean dircheck libs $(xshell_exe)
$(xshell_exe): $(xshell_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(util_dir),xshell,$(xshell_obj))

# -- basictest --

.PHONY : basictest
basictest: status clean dircheck $(test_dir)/basictest$(exe_suffix)
$(test_dir)/basictest$(exe_suffix): $(ut_basictest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),basictest,$(ut_basictest_obj))

# -- binarytest --

.PHONY : binarytest
binarytest: status clean dircheck $(test_dir)/binarytest$(exe_suffix)
$(test_dir)/binarytest$(exe_suffix): $(ut_binarytest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),binarytest,$(ut_binarytest_obj))

# -- colldeftest --

.PHONY : colldeftest
colldeftest: status clean dircheck $(test_dir)/colldeftest$(exe_suffix)
$(test_dir)/colldeftest$(exe_suffix): $(ut_colldeftest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),colldeftest,$(ut_colldeftest_obj))

# -- dictchangetest --

.PHONY : dictchangetest
dictchangetest: status clean dircheck $(test_dir)/dictchangetest$(exe_suffix)
$(test_dir)/dictchangetest$(exe_suffix): $(ut_dictchangetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),dictchangetest,$(ut_dictchangetest_obj))

# -- dictdeftest --

.PHONY : dictdeftest
dictdeftest: status clean dircheck $(test_dir)/dictdeftest$(exe_suffix)
$(test_dir)/dictdeftest$(exe_suffix): $(ut_dictdeftest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),dictdeftest,$(ut_dictdeftest_obj))

# -- dirtyexittest1 --

.PHONY : dirtyexittest1
dirtyexittest1: status clean dircheck $(test_dir)/dirtyexittest1$(exe_suffix)
$(test_dir)/dirtyexittest1$(exe_suffix): $(ut_dirtyexittest1_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),dirtyexittest1,$(ut_dirtyexittest1_obj))

# -- dirtyexittest2 --

.PHONY : dirtyexittest2
dirtyexittest2: status clean dircheck $(test_dir)/dirtyexittest2$(exe_suffix)
$(test_dir)/dirtyexittest2$(exe_suffix): $(ut_dirtyexittest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),dirtyexittest2,$(ut_dirtyexittest2_obj))

# -- domnodetest --

.PHONY : domnodetest
domnodetest: status clean dircheck $(test_dir)/domnodetest$(exe_suffix)
$(test_dir)/domnodetest$(exe_suffix): $(ut_domnodetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),domnodetest,$(ut_domnodetest_obj))

# -- enctest --

.PHONY : enctest
enctest: status clean dircheck $(test_dir)/enctest$(exe_suffix)
$(test_dir)/enctest$(exe_suffix): $(ut_enctest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),enctest,$(ut_enctest_obj))

# -- importtest --

.PHONY : importtest
importtest: status clean dircheck $(test_dir)/importtest$(exe_suffix)
$(test_dir)/importtest$(exe_suffix): $(ut_importtest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),importtest,$(ut_importtest_obj))

# -- indexdeftest --

.PHONY : indexdeftest
indexdeftest: status clean dircheck $(test_dir)/indexdeftest$(exe_suffix)
$(test_dir)/indexdeftest$(exe_suffix): $(ut_indexdeftest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),indexdeftest,$(ut_indexdeftest_obj))

# -- indextest1 --

.PHONY : indextest1
indextest1: status clean dircheck $(test_dir)/indextest1$(exe_suffix)
$(test_dir)/indextest1$(exe_suffix): $(ut_indextest1_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),indextest1,$(ut_indextest1_obj))

# -- indextest2 --

.PHONY : indextest2
indextest2: status clean dircheck $(test_dir)/indextest2$(exe_suffix)
$(test_dir)/indextest2$(exe_suffix): $(ut_indextest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),indextest2,$(ut_indextest2_obj))

# -- indextest3 --

.PHONY : indextest3
indextest3: status clean dircheck $(test_dir)/indextest3$(exe_suffix)
$(test_dir)/indextest3$(exe_suffix): $(ut_indextest3_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),indextest3,$(ut_indextest3_obj))

# -- metaphonetest --

.PHONY : metaphonetest
metaphonetest: status clean dircheck $(test_dir)/metaphonetest$(exe_suffix)
$(test_dir)/metaphonetest$(exe_suffix): $(ut_metaphonetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),metaphonetest,$(ut_metaphonetest_obj))

# -- namespacetest --

.PHONY : namespacetest
namespacetest: status clean dircheck $(test_dir)/namespacetest$(exe_suffix)
$(test_dir)/namespacetest$(exe_suffix): $(ut_namespacetest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),namespacetest,$(ut_namespacetest_obj))

# -- regressiontest --

.PHONY : regressiontest
regressiontest: status clean dircheck $(test_dir)/regressiontest$(exe_suffix)
$(test_dir)/regressiontest$(exe_suffix): $(ut_regressiontest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),regressiontest,$(ut_regressiontest_obj))

# -- rfltest --

.PHONY : rfltest
rfltest: status clean dircheck $(test_dir)/rfltest$(exe_suffix)
$(test_dir)/rfltest$(exe_suffix): $(ut_rfltest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),rfltest,$(ut_rfltest_obj))

# -- sortkeytest --

.PHONY : sortkeytest
sortkeytest: status clean dircheck $(test_dir)/sortkeytest$(exe_suffix)
$(test_dir)/sortkeytest$(exe_suffix): $(ut_sortkeytest_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),sortkeytest,$(ut_sortkeytest_obj))

# -- sortkeytest2 --

.PHONY : sortkeytest2
sortkeytest2: status clean dircheck $(test_dir)/sortkeytest2$(exe_suffix)
$(test_dir)/sortkeytest2$(exe_suffix): $(ut_sortkeytest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),sortkeytest2,$(ut_sortkeytest2_obj))

# -- xpathtest1 --

.PHONY : xpathtest1
xpathtest1: status clean dircheck $(test_dir)/xpathtest1$(exe_suffix)
$(test_dir)/xpathtest1$(exe_suffix): $(ut_xpathtest1_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),xpathtest1,$(ut_xpathtest1_obj))

# -- xpathtest2 --

.PHONY : xpathtest2
xpathtest2: status clean dircheck $(test_dir)/xpathtest2$(exe_suffix)
$(test_dir)/xpathtest2$(exe_suffix): $(ut_xpathtest2_obj) $(xflaim_static_lib)
	$(ec)$(gprintf) "Linking $@ ...\n"
	$(call flm_exe_link_cmd,$(test_dir),xpathtest2,$(ut_xpathtest2_obj))

# -- java (JNI jar file) --

.PHONY : java
java: status clean dircheck $(xflaim_jni_jar) $(xflaim_shared_lib)

.PHONY : xedit
xedit: java $(xedit_java_exe)

# -- csharp

.PHONY : csharp
csharp: status clean dircheck $(xflaim_csharp_assembly) $(xflaim_shared_lib)

.PHONY : cstest
cstest: csharp $(cstest_exe)

# -- version --

define make_version_files
	$(ec)$(gprintf) "$(version)" > $(1)/VERSION
	$(ec)$(gprintf) " " > $(1)/SVNRevision.$(svn_revision)
	$(ec)$(gprintf) "Version files created.\n"
endef

# -- srcdist --

.PHONY : srcdist
srcdist: status clean dircheck docs spec
   ifeq "$(svn_revision)" "0"
		$(error SVN revision cannot be $(svn_revision))
   else
		$(ec)$(gprintf) "Creating source package (SVN revision $(svn_revision)) ...\n"
   endif
	-$(ec)$(call rmdircmd,$(package_stage_dir))
	$(ec)$(call mkdircmd,$(package_stage_dir))
	$(ec)$(call make_version_files,$(package_stage_dir)) 
	$(ec)$(call copycmd,Makefile,$(package_stage_dir))
	$(ec)$(call copycmd,COPYING,$(package_stage_dir))
	$(ec)$(call copycmd,COPYRIGHT,$(package_stage_dir))
	$(ec)$(call dircopycmd,docs,$(package_stage_dir)/docs)
	$(ec)$(call dircopycmd,src,$(package_stage_dir)/src)
	$(ec)$(call copycmd,$(ftk_src_dir)/ftk.h,$(package_stage_dir)/src/xflaimtk.h)
	$(ec)$(call dircopycmd,util,$(package_stage_dir)/util)
	$(ec)$(call dircopycmd,debian,$(package_stage_dir)/debian)
	$(ec)$(call dircopycmd,sample,$(package_stage_dir)/sample)
	$(ec)$(call dircopycmd,csharp,$(package_stage_dir)/csharp)
	$(ec)$(call dircopycmd,java,$(package_stage_dir)/java)
	$(ec)$(call dircopycmd,$(doxygen_output_dir),$(package_stage_dir)/docs)
ifdef jhome	
	$(ec)$(call dircopycmd,$(javadoc_output_dir),$(package_stage_dir)/javadocs)
endif
ifeq ($(host_os_family),win)
	$(ec)$(call dircopycmd,$(dir $(topdir))tools,$(package_stage_dir)/tools)
endif
	$(ec)$(call mkdircmd,$(package_stage_dir)/ftk)
	$(ec)$(call dircopycmd,$(dir $(topdir))ftk/src,$(package_stage_dir)/ftk/src)
ifneq ($(host_os_family),win)
	-$(ec)rm -rf `find $(package_stage_dir) -name .svn`
endif
ifeq ($(host_os_family),win)
	$(ec)$(call copycmd,make.exe,$(package_stage_dir))
endif
	$(ec)$(call create_archive,$(package_stage_parent_dir), \
		$(src_package_dir)/$(src_package_base_name), \
		$(package_proj_name_and_ver))
	$(ec)$(call rmdircmd,$(package_stage_parent_dir))
	$(ec)$(gprintf) "Source package created.\n"

# -- bindist --

.PHONY : bindist
bindist: status clean dircheck all binpackage
	$(ec)$(gprintf) ""

# -- binpackage --

.PHONY : binpackage
binpackage: status
   ifeq "$(svn_revision)" "0"
		$(error SVN revision cannot be $(svn_revision))
   else
		$(ec)$(gprintf) "Creating binary package (SVN revision $(svn_revision)) ...\n"
   endif
	-$(ec)$(call rmdircmd,$(package_stage_dir))
	$(ec)$(call mkdircmd,$(package_stage_dir))
	$(ec)$(call mkdircmd,$(package_inc_stage_dir))
	$(ec)$(call mkdircmd,$(package_shared_lib_stage_dir))
	$(ec)$(call mkdircmd,$(package_static_lib_stage_dir))
	$(ec)$(call mkdircmd,$(package_util_stage_dir))
	$(ec)$(call make_version_files,$(package_stage_dir)) 
	$(ec)$(call copycmd,COPYING,$(package_stage_dir))
	$(ec)$(call copycmd,COPYRIGHT,$(package_stage_dir))
	$(ec)$(call copycmd,src/xflaim.h,$(package_inc_stage_dir))
	$(ec)$(call copycmd,$(ftk_src_dir)/ftk.h,$(package_inc_stage_dir)/xflaimtk.h)
	$(ec)$(call copycmd,$(xflaim_static_lib),$(package_static_lib_stage_dir))
ifdef xflaim_shared_lib
	$(ec)$(call copycmd,$(xflaim_shared_lib),$(package_shared_lib_stage_dir))
endif
ifdef win_target
	$(ec)$(call copycmd,$(xflaim_shared_imp_lib),$(package_shared_lib_stage_dir))
endif
	$(ec)$(call copycmd,$(checkdb_exe),$(package_util_stage_dir))
	$(ec)$(call copycmd,$(rebuild_exe),$(package_util_stage_dir))
	$(ec)$(call copycmd,$(view_exe),$(package_util_stage_dir))
	$(ec)$(call copycmd,$(xshell_exe),$(package_util_stage_dir))
	$(ec)$(call create_archive,$(package_stage_parent_dir), \
		$(bin_package_dir)/$(bin_package_base_name), \
		$(package_proj_name_and_ver))
	$(ec)$(call rmdircmd,$(package_stage_parent_dir))
	$(ec)$(gprintf) "Binary package created.\n"

# -- dist --

.PHONY : dist
dist: status clean dircheck srcdist
   ifeq "$(svn_revision)" "0"
		$(error SVN revision cannot be $(svn_revision))
   else
		$(ec)$(gprintf) "Creating distribution (SVN revision $(svn_revision)) ...\n"
   endif
	$(ec)$(call copycmd,$(src_package_dir)/$(src_package_name),$(package_dir))
	$(ec)$(call extract_archive,$(package_dir),$(src_package_base_name))
	$(ec)$(MAKE) -C $(package_dir)/$(package_proj_name_and_ver) clean
	$(ec)$(MAKE) -C $(package_dir)/$(package_proj_name_and_ver) $(submake_targets) all
	$(ec)$(MAKE) -C $(package_dir)/$(package_proj_name_and_ver) $(submake_targets) binpackage package_dir="$(package_dir)"
	$(ec)$(call rmdircmd,$(package_dir)/$(package_proj_name_and_ver))
	$(ec)$(call rmcmd,$(package_dir)/$(src_package_name))
	$(ec)$(gprintf) "Distribution created.\n"

# -- Change log --

.PHONY : changelog
changelog:
	$(ec)$(gprintf) "Creating change log for SVN revisions $(svn_low_rev)-$(svn_high_rev) ...\n"
	$(ec)$(gprintf) "Using SVN URL $(svnurl) ...\n"
	$(ec)svn log $(svnurl) -v -r $(svn_low_rev):$(svn_high_rev) > $(package_sources_dir)/$(package_proj_name_and_ver).tar.log
	$(ec)$(gprintf) "Change log created.\n"

# -- install --

.PHONY : install
install: libs pkgconfig
ifneq ($(host_os_family),win)
	$(ec)$(gprintf) "Installing ...\n"
	mkdir -p $(lib_install_dir)/pkgconfig
	mkdir -p $(include_install_dir)
	install --mode=755 $(xflaim_shared_lib) $(lib_install_dir)
	install --mode=644 $(xflaim_static_lib) $(lib_install_dir)
	install --mode=755 $(xflaim_csharp_assembly) $(lib_install_dir)
	install --mode=644 $(xflaim_csharp_assembly_xml) $(lib_install_dir)
	install --mode=644 $(pkgconfig_file) $(pkgconfig_install_dir)
	install --mode=644 src/xflaim.h $(include_install_dir)
	install --mode=644 $(ftk_src_dir)/ftk.h $(include_install_dir)/xflaimtk.h
ifneq ($(so_age),0)
ifneq ($(so_revision),0)
	cd $(lib_install_dir); ln -fs $(lib_prefix)$(project_name).so.$(so_current).$(so_revision).$(so_age) $(lib_prefix)$(project_name).so.$(so_current).$(so_revision) 
endif
endif
ifneq ($(so_revision),0)
	cd $(lib_install_dir); ln -fs $(lib_prefix)$(project_name).so.$(so_current).$(so_revision) $(lib_prefix)$(project_name).so.$(so_current) 
endif
	cd $(lib_install_dir); ln -fs $(lib_prefix)$(project_name).so.$(so_current) $(lib_prefix)$(project_name).so
	-ldconfig $(lib_install_dir)
	$(ec)$(gprintf) "Installation complete.\n"
endif

# -- uninstall --

.PHONY : uninstall
uninstall: 
ifneq ($(host_os_family),win)
	$(ec)$(gprintf) "Uninstalling ...\n"
ifneq ($(so_age),0)
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name).so.$(so_current).$(so_revision).$(so_age)
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name).so.$(so_current).$(so_revision)
else
ifneq ($(so_revision),0)
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name).so.$(so_current).$(so_revision)
endif
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name).so.$(so_current)
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name).so
endif
	-rm -rf $(lib_install_dir)/$(lib_prefix)$(project_name)$(static_lib_suffix)
	-rm -rf $(pkgconfig_install_dir)/$(pkgconfig_file_name)
	-rm -rf $(include_install_dir)/xflaim.h
	-rm -rf $(include_install_dir)/xflaimtk.h
	$(ec)$(gprintf) "Uninstalled.\n"
endif

# -- spec file --

.PHONY : spec
spec: dircheck
	$(ec)$(gprintf) "Creating spec file ...\n"
	$(ec)$(gprintf) "Name: $(package_proj_name)\n" > $(spec_file)
	$(ec)$(gprintf) "$(percent)define prefix $(install_prefix)\n" >> $(spec_file)
	$(ec)$(gprintf) "BuildRequires: gcc-c++ java2-devel-packages libstdc++ libstdc++-devel mono-devel ncurses-devel\n" >> $(spec_file)
	$(ec)$(gprintf) "Summary: $(project_brief_desc)\n" >> $(spec_file)
	$(ec)$(gprintf) "URL: http://forge.novell.com/modules/xfmod/project/$(question)flaim\n" >> $(spec_file)
	$(ec)$(gprintf) "Version: $(version)\n" >> $(spec_file)
	$(ec)$(gprintf) "Release: $(package_release_num)\n" >> $(spec_file)
	$(ec)$(gprintf) "License: GNU Library General Public License v. 2.0 and 2.1 (LGPL)\n" >> $(spec_file)
	$(ec)$(gprintf) "Vendor: Novell, Inc.\n" >> $(spec_file)
	$(ec)$(gprintf) "Group: Development/Libraries/C and C++\n" >> $(spec_file)
	$(ec)$(gprintf) "Source: $(package_proj_name_and_ver).tar.gz\n" >> $(spec_file)
	$(ec)$(gprintf) "BuildRoot: $(percent){_tmppath}/$(percent){name}-$(percent){version}-build\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)description\n" >> $(spec_file)
	$(ec)$(gprintf) "XFLAIM is an embeddable cross-platform XML database engine that\n" >> $(spec_file)
	$(ec)$(gprintf) "provides a rich, powerful, easy-to-use feature set. It has proven to be\n" >> $(spec_file)
	$(ec)$(gprintf) "highly scalable, reliable, and robust. It is available on a wide\n" >> $(spec_file)
	$(ec)$(gprintf) "variety of 32 bit and 64 bit platforms.\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "Authors:\n" >> $(spec_file)
	$(ec)$(gprintf) "$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)\n" >> $(spec_file)
	$(ec)$(gprintf) "    $(dsanders_info)\n" >> $(spec_file)
	$(ec)$(gprintf) "    $(ahodgkinson_info)\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)package devel\n" >> $(spec_file)
	$(ec)$(gprintf) "Summary: $(project_brief_desc)\n" >> $(spec_file)
	$(ec)$(gprintf) "Group: Development/Libraries/C and C++\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)description devel\n" >> $(spec_file)
	$(ec)$(gprintf) "XFLAIM is an embeddable cross-platform XML database engine that\n" >> $(spec_file)
	$(ec)$(gprintf) "provides a rich, powerful, easy-to-use feature set. It has proven to be\n" >> $(spec_file)
	$(ec)$(gprintf) "highly scalable, reliable, and robust. It is available on a wide\n" >> $(spec_file)
	$(ec)$(gprintf) "variety of 32 bit and 64 bit platforms.\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "Authors:\n" >> $(spec_file)
	$(ec)$(gprintf) "$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)$(dash)\n" >> $(spec_file)
	$(ec)$(gprintf) "    Daniel Sanders\n" >> $(spec_file)
	$(ec)$(gprintf) "    Andrew Hodgkinson\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)prep\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)setup -q\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)build\n" >> $(spec_file)
	$(ec)$(gprintf) "$(MAKE) lib_dir_name=$(percent){_lib} $(submake_targets) libs csharp\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)install\n" >> $(spec_file)
	$(ec)$(gprintf) "$(MAKE) rpm_build_root=$(dollar)RPM_BUILD_ROOT install_prefix=$(percent){prefix} lib_dir_name=$(percent){_lib} $(submake_targets) install\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)clean\n" >> $(spec_file)
	$(ec)$(gprintf) "rm -rf $(dollar)RPM_BUILD_ROOT\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)files\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)defattr(-,root,root)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)doc COPYING COPYRIGHT VERSION\n" >> $(spec_file)
ifneq ($(so_age),0)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name).so.$(so_current).$(so_revision).$(so_age)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name).so.$(so_current).$(so_revision)\n" >> $(spec_file)
else
ifneq ($(so_revision),0)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name).so.$(so_current).$(so_revision)\n" >> $(spec_file)
endif
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name).so.$(so_current)\n" >> $(spec_file)
endif
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/xflaim_csharp.dll\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/xflaim_csharp.xml\n" >> $(spec_file)
	$(ec)$(gprintf) "\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)files devel\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent)defattr(-,root,root)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name).so\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/$(lib_prefix)$(project_name)$(static_lib_suffix)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/$(percent){_lib}/pkgconfig/$(pkgconfig_file_name)\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/include/xflaim.h\n" >> $(spec_file)
	$(ec)$(gprintf) "$(percent){prefix}/include/xflaimtk.h\n" >> $(spec_file)
	$(ec)$(gprintf) "Created spec file.\n"

# -- PKG-CONFIG --

.PHONY : pkgconfig
pkgconfig: dircheck
	$(ec)$(gprintf) "prefix=$(install_prefix)\n" > $(pkgconfig_file)
	$(ec)$(gprintf) "exec_prefix=$(dollar){prefix}\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "libdir=$(dollar){exec_prefix}/$(lib_dir_name)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "includedir=$(dollar){prefix}/include\n\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Name: $(package_proj_name)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Description: $(project_brief_desc)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Version: $(version)\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Libs: $(link_libs) -lxflaim -L$(dollar){libdir}\n" >> $(pkgconfig_file)
	$(ec)$(gprintf) "Cflags: -I$(dollar){includedir}\n" >> $(pkgconfig_file)

# -- SRCRPM --

.PHONY : srcrpm
srcrpm: srcdist spec
	$(ec)$(gprintf) "Creating source RPM ...\n"
	$(ec)rpmbuild --define="_topdir $(package_dir)" --quiet --nodeps -bs $(spec_file)
	$(ec)$(gprintf) "Source RPM created.\n"

# -- RPMS --

.PHONY : rpms
rpms: spec srcdist
	$(ec)$(gprintf) "Creating source and binary RPMs ...\n"
	$(ec)rpmbuild --define="_topdir $(package_dir)" --quiet --nodeps -ba $(spec_file)
	$(ec)find $(package_dir) -name *.rpm | xargs chmod 775
	$(ec)$(gprintf) "Source and binary RPMs created.\n"

# -- Ubuntu Source Package --

.PHONY : ubuntusrc
ubuntusrc: srcdist
	$(ec)$(gprintf) "Creating Ubuntu source package ...\n"
	
	-$(ec)$(call rmdircmd,$(ubuntu_stage_dir))
	$(ec)$(call mkdircmd,$(ubuntu_stage_dir))
	$(ec)$(call copycmd,$(src_package_dir)/$(src_package_name),$(ubuntu_stage_dir)/$(src_package_base_name).orig.tar.gz)
	$(ec)$(call extract_archive,$(ubuntu_stage_dir),$(src_package_base_name).orig)
	$(ec)$(call rmcmd,$(ubuntu_stage_dir)/$(src_package_base_name).orig.tar)

	$(ec)$(call copycmd,$(ubuntu_stage_dir)/$(package_proj_name_and_ver)/COPYRIGHT,$(ubuntu_stage_dir)/$(package_proj_name_and_ver)/debian/copyright)
	$(ec)$(call copycmd,$(src_package_dir)/$(src_package_name),$(ubuntu_stage_dir)/$(package_proj_name)_$(version).orig.tar.gz)

	$(ec)$(gprintf) "Creating Ubuntu changelog file ...\n"
	$(ec)$(gprintf) "$(package_proj_name) ($(package_version_ubuntu)) $(package_distro_ubuntu); urgency=low\n\n" > $(ubuntu_stage_dir)/$(package_proj_name_and_ver)/debian/changelog
	$(ec)$(gprintf) "  * Package update for Ubuntu.\n\n" >> $(ubuntu_stage_dir)/$(package_proj_name_and_ver)/debian/changelog
	$(ec)$(gprintf) " -- Andrew Hodgkinson (Sr. Software Engineer) <ahodgkinson@novell.com>  " >> $(ubuntu_stage_dir)/$(package_proj_name_and_ver)/debian/changelog
	$(ec)822-date >> $(ubuntu_stage_dir)/$(package_proj_name_and_ver)/debian/changelog
	$(ec)$(gprintf) "\n" >> $(ubuntu_stage_dir)/$(package_proj_name_and_ver)/debian/changelog
	$(ec)cat ChangeLog.ubuntu >> $(ubuntu_stage_dir)/$(package_proj_name_and_ver)/debian/changelog
	
	$(ec)cd $(ubuntu_stage_dir)/$(package_proj_name_and_ver); dpkg-buildpackage -S -sa -rfakeroot
	
	$(ec)$(gprintf) "Checking Ubuntu package ...\n"
	$(ec)lintian -i $(ubuntu_stage_dir)/$(package_proj_name)_$(package_version_ubuntu).dsc
	$(ec)linda -i $(ubuntu_stage_dir)/$(package_proj_name)_$(package_version_ubuntu).dsc
	
	$(ec)$(gprintf) "Moving packages to UBUNTU directory ...\n"
	$(ec)$(call copycmd,$(ubuntu_stage_dir)/*.dsc,$(package_ubuntu_dir))
	$(ec)$(call rmcmd,$(ubuntu_stage_dir)/*.dsc)
	$(ec)$(call copycmd,$(ubuntu_stage_dir)/*.tar.gz,$(package_ubuntu_dir))
	$(ec)$(call rmcmd,$(ubuntu_stage_dir)/*.tar.gz)
	$(ec)$(call copycmd,$(ubuntu_stage_dir)/*.changes,$(package_ubuntu_dir))
	$(ec)$(call rmcmd,$(ubuntu_stage_dir)/*.changes)

	$(ec)$(gprintf) "Removing temporary files ...\n"
	$(ec)$(call rmdircmd,$(ubuntu_stage_dir))
	
	$(ec)$(gprintf) "Done.\n"
	
# -- Documentation --

.PHONY : docs
docs: status clean dircheck doxyfile
	$(ec)$(gprintf) "Creating documentation ...\n"
	$(ec)doxygen $(doxyfile)
ifdef jhome
	$(ec)$(gprintf) "Creating JAVADOC documentation ...\n"
	$(ec)$(call hostpath,$(jhome)/bin/javadoc) -sourcepath java -d $(call hostpath,$(javadoc_output_dir)) $(project_name)
endif
	$(ec)$(gprintf) "Documentation created.\n"

# -- misc. targets --

.PHONY : dircheck
dircheck:
	$(ec)$(call mkdircmd,$(util_obj_dir))
	$(ec)$(call mkdircmd,$(test_obj_dir))
	$(ec)$(call mkdircmd,$(sample_obj_dir))
	$(ec)$(call mkdircmd,$(lib_obj_dir))
	$(ec)$(call mkdircmd,$(csharp_obj_dir))
	$(ec)$(call mkdircmd,$(java_output_dir))
	$(ec)$(call mkdircmd,$(java_class_output_dir))
	$(ec)$(call mkdircmd,$(csharp_output_dir))
	$(ec)$(call mkdircmd,$(cstest_output_dir))
	$(ec)$(call mkdircmd,$(doxygen_output_dir))
	$(ec)$(call mkdircmd,$(javadoc_output_dir))
	$(ec)$(call mkdircmd,$(util_dir))
	$(ec)$(call mkdircmd,$(test_dir))
	$(ec)$(call mkdircmd,$(sample_dir))
	$(ec)$(call mkdircmd,$(static_lib_dir))
	$(ec)$(call mkdircmd,$(shared_lib_dir))
	$(ec)$(call mkdircmd,$(package_dir))
	$(ec)$(call mkdircmd,$(spec_dir))
	$(ec)$(call mkdircmd,$(package_sources_dir))
	$(ec)$(call mkdircmd,$(package_bin_dir))
	$(ec)$(call mkdircmd,$(package_build_dir))
	$(ec)$(call mkdircmd,$(package_rpms_dir))
	$(ec)$(call mkdircmd,$(package_srpms_dir))
	$(ec)$(call mkdircmd,$(package_debian_dir))
	$(ec)$(call mkdircmd,$(package_ubuntu_dir))

# -- phony targets --

.PHONY : all
all: libs allutils
	$(ec)$(gprintf) ""

.PHONY : allutils
allutils: status dircheck libs $(util_targets)
	$(ec)$(gprintf) ""

.PHONY : test
test:	status dircheck $(xflaim_static_lib) $(test_targets)
ifndef netware_target
	$(ec)$(call copycmd,util/xmlfiles/*.xml,$(test_dir))
	$(ec)$(call runtest,basictest)
	$(ec)$(call runtest,binarytest)
	$(ec)$(call runtest,colldeftest)
	$(ec)$(call runtest,dictchangetest)
	$(ec)$(call runtest,dictdeftest)
	$(ec)$(call runtest,dirtyexittest1)
	$(ec)$(call runtest,dirtyexittest2)
	$(ec)$(call runtest,domnodetest)
	$(ec)$(call runtest,enctest)
	$(ec)$(call runtest,importtest)
	$(ec)$(call runtest,indexdeftest)
	$(ec)$(call runtest,indextest1)
	$(ec)$(call runtest,indextest2)
	$(ec)$(call runtest,indextest3)
	$(ec)$(call runtest,namespacetest)
	$(ec)$(call runtest,metaphonetest)
	$(ec)$(call runtest,regressiontest)
	$(ec)$(call runtest,rfltest)
	$(ec)$(call runtest,sortkeytest)
	$(ec)$(call runtest,sortkeytest2)
	$(ec)$(call runtest,xpathtest1)
	$(ec)$(call runtest,xpathtest2)
endif

.PHONY : debug
debug:
	$(ec)$(gprintf) ""

.PHONY : release
release:
	$(ec)$(gprintf) ""

.PHONY : usegcc
usegcc:
	$(ec)$(gprintf) ""

.PHONY : 32bit
32bit:
	$(ec)$(gprintf) ""

.PHONY : 64bit
64bit:
	$(ec)$(gprintf) ""

.PHONY : win
win:
	$(ec)$(gprintf) ""

.PHONY : linux
linux:
	$(ec)$(gprintf) ""

.PHONY : solaris
solaris:
	$(ec)$(gprintf) ""

.PHONY : sparcgeneric
sparcgeneric:
	$(ec)$(gprintf) ""

.PHONY : osx
osx:
	$(ec)$(gprintf) ""

.PHONY : nlm
nlm:
	$(ec)$(gprintf) ""

.PHONY : ring0
ring0:
	$(ec)$(gprintf) ""
	
.PHONY : openssl
openssl:
	$(ec)$(gprintf) ""
	
.PHONY : verbose
verbose:
	$(ec)$(gprintf) ""

.PHONY : check
check:
	$(ec)$(gprintf) ""

.PHONY : TAGS
TAGS:
	$(ec)$(gprintf) ""

.PHONY : info
info:
	$(ec)$(gprintf) ""

.PHONY : ignore-local-mods
ignore-local-mods:
	$(ec)$(gprintf) ""

.PHONY : ilm
ilm:
	$(ec)$(gprintf) ""

.PHONY : installcheck
installcheck:
	$(ec)$(gprintf) ""

.PHONY : clean
clean:
ifeq ($(do_clean),1)
	$(ec)$(gprintf) "\n"
	$(ec)$(gprintf) "Cleaning $(target_path) ...\n"
	-$(ec)$(call rmdircmd,$(target_path))
	-$(ec)$(call rmcmd,*.pch)
	$(ec)$(gprintf) "\n"
endif

.PHONY : distclean
	-$(ec)$(call rmcmd,*.pch)

.PHONY : mostlyclean
mostlyclean : clean
	$(ec)$(gprintf) ""

.PHONY : maintainer-clean
maintainer-clean:
	-$(ec)$(call rmdircmd,$(build_output_dir))
	-$(ec)$(call rmcmd,*.pch)

.PHONY : status
status:
	$(ec)$(gprintf) "===============================================================================\n"
	$(ec)$(gprintf) "SVN Revision.................... $(svn_revision)\n"
	$(ec)$(gprintf) "Host Operating System Family.... $(host_os_family)\n"
	$(ec)$(gprintf) "Top Directory................... $(call ppath,$(topdir))\n"
	$(ec)$(gprintf) "Target Operating System Family.. $(target_os_family)\n"
	$(ec)$(gprintf) "Target Processor Family......... $(target_processor_family)\n"
	$(ec)$(gprintf) "Target Word Size................ $(target_word_size)\n"
	$(ec)$(gprintf) "Target Build Type............... $(target_build_type)\n"
	$(ec)$(gprintf) "Target Path..................... $(call ppath,$(target_path))\n"
	$(ec)$(gprintf) "Toolkit Path.................... $(call ppath,$(ftk_dir))\n"
	$(ec)$(gprintf) "Install Prefix.................. $(call ppath,$(install_prefix))\n"
	$(ec)$(gprintf) "Compiler........................ $(call ppath,$(compiler))\n"
	$(ec)$(gprintf) "Librarian....................... $(call ppath,$(libr))\n"
	$(ec)$(gprintf) "JAVA HOME....................... $(call ppath,$(jhome))\n"
	$(ec)$(gprintf) "Defines......................... $(strip $(ccdefs))\n"
	$(ec)$(gprintf) "===============================================================================\n"

.PHONY : doxyfile
doxyfile: dircheck
	$(ec)$(gprintf) "PROJECT_NAME           = \"$(project_display_name)\"\n" > $(doxyfile)
	$(ec)$(gprintf) "PROJECT_NUMBER         = \"$(version)\"\n" >> $(doxyfile)
	$(ec)$(gprintf) "OUTPUT_DIRECTORY       = $(doxygen_output_dir)\n" >> $(doxyfile)
	$(ec)$(gprintf) "CREATE_SUBDIRS         = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "OUTPUT_LANGUAGE        = English\n" >> $(doxyfile)
	$(ec)$(gprintf) "USE_WINDOWS_ENCODING   = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "BRIEF_MEMBER_DESC      = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "REPEAT_BRIEF           = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "ABBREVIATE_BRIEF       = \"The $(dollar)name class\" $(backslash)\n" >> $(doxyfile)
	$(ec)$(gprintf) "                         \"The $(dollar)name widget\" $(backslash)\n" >> $(doxyfile)
	$(ec)$(gprintf) "                         \"The $(dollar)name file\" $(backslash)\n" >> $(doxyfile)
	$(ec)$(gprintf) "                         is $(backslash)\n" >> $(doxyfile)
	$(ec)$(gprintf) "                         provides $(backslash)\n" >> $(doxyfile)
	$(ec)$(gprintf) "                         specifies $(backslash)\n" >> $(doxyfile)
	$(ec)$(gprintf) "                         contains $(backslash)\n" >> $(doxyfile)
	$(ec)$(gprintf) "                         represents $(backslash)\n" >> $(doxyfile)
	$(ec)$(gprintf) "                         a $(backslash)\n" >> $(doxyfile)
	$(ec)$(gprintf) "                         an $(backslash)\n" >> $(doxyfile)
	$(ec)$(gprintf) "                         the\n" >> $(doxyfile)
	$(ec)$(gprintf) "ALWAYS_DETAILED_SEC    = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "INLINE_INHERITED_MEMB  = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "FULL_PATH_NAMES        = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "STRIP_FROM_PATH        = \"\"\n" >> $(doxyfile)
	$(ec)$(gprintf) "STRIP_FROM_INC_PATH    = \n" >> $(doxyfile)
	$(ec)$(gprintf) "SHORT_NAMES            = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "JAVADOC_AUTOBRIEF      = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "MULTILINE_CPP_IS_BRIEF = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "DETAILS_AT_TOP         = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "INHERIT_DOCS           = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "SEPARATE_MEMBER_PAGES  = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "TAB_SIZE               = 3\n" >> $(doxyfile)
	$(ec)$(gprintf) "ALIASES                = \n" >> $(doxyfile)
	$(ec)$(gprintf) "OPTIMIZE_OUTPUT_FOR_C  = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "OPTIMIZE_OUTPUT_JAVA   = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "BUILTIN_STL_SUPPORT    = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "DISTRIBUTE_GROUP_DOC   = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "SUBGROUPING            = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "EXTRACT_ALL            = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "EXTRACT_PRIVATE        = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "EXTRACT_STATIC         = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "EXTRACT_LOCAL_CLASSES  = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "EXTRACT_LOCAL_METHODS  = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "HIDE_UNDOC_MEMBERS     = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "HIDE_UNDOC_CLASSES     = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "HIDE_FRIEND_COMPOUNDS  = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "HIDE_IN_BODY_DOCS      = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "INTERNAL_DOCS          = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "CASE_SENSE_NAMES       = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "HIDE_SCOPE_NAMES       = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "SHOW_INCLUDE_FILES     = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "INLINE_INFO            = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "SORT_MEMBER_DOCS       = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "SORT_BRIEF_DOCS        = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "SORT_BY_SCOPE_NAME     = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_TODOLIST      = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_TESTLIST      = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_BUGLIST       = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_DEPRECATEDLIST= YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "ENABLED_SECTIONS       = \n" >> $(doxyfile)
	$(ec)$(gprintf) "MAX_INITIALIZER_LINES  = 30\n" >> $(doxyfile)
	$(ec)$(gprintf) "SHOW_USED_FILES        = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "SHOW_DIRECTORIES       = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "FILE_VERSION_FILTER    = \n" >> $(doxyfile)
	$(ec)$(gprintf) "QUIET                  = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "WARNINGS               = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "WARN_IF_UNDOCUMENTED   = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "WARN_IF_DOC_ERROR      = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "WARN_NO_PARAMDOC       = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "WARN_FORMAT            = \"$(dollar)file:$(dollar)line: $(dollar)text\"\n" >> $(doxyfile)
	$(ec)$(gprintf) "WARN_LOGFILE           = \n" >> $(doxyfile)
	$(ec)$(gprintf) "INPUT                  = src/xflaim.h $(backslash)\n" >> $(doxyfile)
	$(ec)$(gprintf) "									$(ftk_src_dir)/ftk.h\n" >> $(doxyfile)
	$(ec)$(gprintf) "FILE_PATTERNS          = *.h\n" >> $(doxyfile)
	$(ec)$(gprintf) "RECURSIVE              = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "EXCLUDE                = \n" >> $(doxyfile)
	$(ec)$(gprintf) "EXCLUDE_SYMLINKS       = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "EXCLUDE_PATTERNS       = \n" >> $(doxyfile)
	$(ec)$(gprintf) "EXAMPLE_PATH           = \n" >> $(doxyfile)
	$(ec)$(gprintf) "EXAMPLE_PATTERNS       = *\n" >> $(doxyfile)
	$(ec)$(gprintf) "EXAMPLE_RECURSIVE      = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "IMAGE_PATH             = \n" >> $(doxyfile)
	$(ec)$(gprintf) "INPUT_FILTER           = \n" >> $(doxyfile)
	$(ec)$(gprintf) "FILTER_PATTERNS        = \n" >> $(doxyfile)
	$(ec)$(gprintf) "FILTER_SOURCE_FILES    = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "SOURCE_BROWSER         = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "INLINE_SOURCES         = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "STRIP_CODE_COMMENTS    = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "REFERENCED_BY_RELATION = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "REFERENCES_RELATION    = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "USE_HTAGS              = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "VERBATIM_HEADERS       = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "ALPHABETICAL_INDEX     = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "COLS_IN_ALPHA_INDEX    = 5\n" >> $(doxyfile)
	$(ec)$(gprintf) "IGNORE_PREFIX          = \n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_HTML          = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "HTML_OUTPUT            = html\n" >> $(doxyfile)
	$(ec)$(gprintf) "HTML_FILE_EXTENSION    = .html\n" >> $(doxyfile)
	$(ec)$(gprintf) "HTML_HEADER            = \n" >> $(doxyfile)
	$(ec)$(gprintf) "HTML_FOOTER            = \n" >> $(doxyfile)
	$(ec)$(gprintf) "HTML_STYLESHEET        = \n" >> $(doxyfile)
	$(ec)$(gprintf) "HTML_ALIGN_MEMBERS     = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_HTMLHELP      = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "CHM_FILE               = \n" >> $(doxyfile)
	$(ec)$(gprintf) "HHC_LOCATION           = \n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_CHI           = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "BINARY_TOC             = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "TOC_EXPAND             = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "DISABLE_INDEX          = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "ENUM_VALUES_PER_LINE   = 4\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_TREEVIEW      = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "TREEVIEW_WIDTH         = 250\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_LATEX         = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "LATEX_OUTPUT           = latex\n" >> $(doxyfile)
	$(ec)$(gprintf) "LATEX_CMD_NAME         = latex\n" >> $(doxyfile)
	$(ec)$(gprintf) "MAKEINDEX_CMD_NAME     = makeindex\n" >> $(doxyfile)
	$(ec)$(gprintf) "COMPACT_LATEX          = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "PAPER_TYPE             = a4wide\n" >> $(doxyfile)
	$(ec)$(gprintf) "EXTRA_PACKAGES         = \n" >> $(doxyfile)
	$(ec)$(gprintf) "LATEX_HEADER           = \n" >> $(doxyfile)
	$(ec)$(gprintf) "PDF_HYPERLINKS         = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "USE_PDFLATEX           = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "LATEX_BATCHMODE        = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "LATEX_HIDE_INDICES     = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_RTF           = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "RTF_OUTPUT             = rtf\n" >> $(doxyfile)
	$(ec)$(gprintf) "COMPACT_RTF            = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "RTF_HYPERLINKS         = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "RTF_STYLESHEET_FILE    = \n" >> $(doxyfile)
	$(ec)$(gprintf) "RTF_EXTENSIONS_FILE    = \n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_MAN           = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "MAN_OUTPUT             = man\n" >> $(doxyfile)
	$(ec)$(gprintf) "MAN_EXTENSION          = .3\n" >> $(doxyfile)
	$(ec)$(gprintf) "MAN_LINKS              = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_XML           = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "XML_OUTPUT             = xml\n" >> $(doxyfile)
	$(ec)$(gprintf) "XML_SCHEMA             = \n" >> $(doxyfile)
	$(ec)$(gprintf) "XML_DTD                = \n" >> $(doxyfile)
	$(ec)$(gprintf) "XML_PROGRAMLISTING     = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_AUTOGEN_DEF   = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_PERLMOD       = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "PERLMOD_LATEX          = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "PERLMOD_PRETTY         = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "PERLMOD_MAKEVAR_PREFIX = \n" >> $(doxyfile)
	$(ec)$(gprintf) "ENABLE_PREPROCESSING   = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "MACRO_EXPANSION        = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "EXPAND_ONLY_PREDEF     = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "SEARCH_INCLUDES        = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "INCLUDE_PATH           = \n" >> $(doxyfile)
	$(ec)$(gprintf) "INCLUDE_FILE_PATTERNS  = \n" >> $(doxyfile)
	$(ec)$(gprintf) "PREDEFINED             = \n" >> $(doxyfile)
	$(ec)$(gprintf) "EXPAND_AS_DEFINED      = \n" >> $(doxyfile)
	$(ec)$(gprintf) "SKIP_FUNCTION_MACROS   = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "TAGFILES               = \n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_TAGFILE       = \n" >> $(doxyfile)
	$(ec)$(gprintf) "ALLEXTERNALS           = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "EXTERNAL_GROUPS        = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "PERL_PATH              = \n" >> $(doxyfile)
	$(ec)$(gprintf) "CLASS_DIAGRAMS         = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "HIDE_UNDOC_RELATIONS   = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "HAVE_DOT               = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "CLASS_GRAPH            = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "COLLABORATION_GRAPH    = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "GROUP_GRAPHS           = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "UML_LOOK               = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "TEMPLATE_RELATIONS     = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "INCLUDE_GRAPH          = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "INCLUDED_BY_GRAPH      = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "CALL_GRAPH             = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "GRAPHICAL_HIERARCHY    = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "DIRECTORY_GRAPH        = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "DOT_IMAGE_FORMAT       = png\n" >> $(doxyfile)
	$(ec)$(gprintf) "DOT_PATH               = \n" >> $(doxyfile)
	$(ec)$(gprintf) "DOTFILE_DIRS           = \n" >> $(doxyfile)
	$(ec)$(gprintf) "MAX_DOT_GRAPH_WIDTH    = 1024\n" >> $(doxyfile)
	$(ec)$(gprintf) "MAX_DOT_GRAPH_HEIGHT   = 1024\n" >> $(doxyfile)
	$(ec)$(gprintf) "MAX_DOT_GRAPH_DEPTH    = 1000\n" >> $(doxyfile)
	$(ec)$(gprintf) "DOT_TRANSPARENT        = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "DOT_MULTI_TARGETS      = NO\n" >> $(doxyfile)
	$(ec)$(gprintf) "GENERATE_LEGEND        = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "DOT_CLEANUP            = YES\n" >> $(doxyfile)
	$(ec)$(gprintf) "SEARCHENGINE           = NO\n" >> $(doxyfile)
