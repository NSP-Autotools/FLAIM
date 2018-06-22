flaim-projects README file
==========================

NOTE: This README file covers platform-independant, and GNU/Linux and
Unix specific information. For information on building and installing
on Windows platforms, please see the README.W32 file.

Contents
--------
1. Project Hierarchy
2. Autotools Build
3. Building RPMs

Project Hierarchy
-----------------

The flaim-projects repository is divided into four sub-projects, named 
for the sub-directories in which they reside:

  * flaim
  * ftk
  * sql
  * xflaim

Each of these sub-projects is a complete project in its own rite. The only
inter-project dependencies among them are that the flaim, sql and xflaim 
projects depend on the FLAIM Tool Kit library (libflaimtk) and header file
(flaimtk.h) provided by the ftk project.

When these four projects are built from the flaim/trunk directory, the 
dependencies are managed for you. 

However, each of the four sub-projects may also be built as separate 
projects, simply by changing into the desired directory, and running
configure/make from that location.

When you build flaim (for instance) by itself in this manner, you need to 
provide the location of the flaimtk library and header file in one of
two ways:

  1. provide the FTKLIB and FTKINC environment variables to configure
  2. create a sym-link within the flaim directory to ../ftk

Both of these techniques work with all three of the higher level projects.

Autotools Build
---------------

IMPORTANT: You must have installed Autoconf version 2.62, Automake version 
1.10 and Libtool version 2.2. These are the latest versions of these three
tools, as of this writing (July 2, 2008).

To build the flaim-projects from a Subversion working copy, just create
a clean working copy from the Subversion repository at:

  https://forgesvn1.novell.com/svn/flaim/trunk flaim-projects

Then enter the following commands from the flaim-projects directory:

  $ autoreconf -i
  $ ./configure && make all check 
 
You may also build from another directory by using a relative path to the
configure script, in this manner:

  $ mkdir build && cd build
  $ ../configure && make all check

To find out what options are available for these packages, you really need to 
check the configure script help page for each of the sub-projects. Most of the
options for each project are the same between all projects, but the ftk project
contains several more options, and the others also have options to help the
build system find the tool kit.

Check the help page for any of the configure scripts like this:

  $ ./configure --help=short

Once you know the options you wish to use, you may use them all on the 
flaim-projects configure script command line. Those not understood by any
given script will simply be ignored.

Building RPMs
-------------

The FLAIM build system provides two custom targets, rpms and srcrpm. These
targets can be used to build RPM packages for RPM-based Linux distributions.

To use these targets, you'll need to be building on an RPM-based platform,
and you'll need to have the platform's package-manager build package installed.
Most development systems have this package installed by default.

If you plan to build RPM's please be aware that the RPM make target (rpms) is 
not quite as automatic as we'd like it to be - however, it's close. The only 
problem you're likely to notice is that you must have the flaim toolkit packages
(flaimtk and flaimtk-devel) installed as RPM packages before you can build the 
flaim, xflaim and sqlflaim RPM packages. As long as the flaimtk ABI doesn't
change too dramatically, you can continue to build against these installed 
packages for future builds (although, you'll probably want to update your 
installed flaimtk packages at reasonable intervals as they are improved).

The easiest way to make this work for you is to configure the ftk directory in
your build tree and then (also from within the ftk directory) type:

  $ make rpms
    ...
  $ sudo rpm -Uvh flaimtk*.rpm
    ...

At this point you will be able to make the rpms target from the top-level 
umbrella project directory, if you wish. Or you may build each individual rpm
package (flaim, xflaim, or sqlflaim) from within their respective project
directories.

Enjoy!

