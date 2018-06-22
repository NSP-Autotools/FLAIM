<?php
/*******************************************************************************
* Desc:	PHP configuration for remote FLAIM builds
* Tabs:	3
*
*		Copyright (c) 2006 Novell, Inc. All Rights Reserved.
*
*		This program is free software; you can redistribute it and/or
*		modify it under the terms of version 2 of the GNU General Public
*		License as published by the Free Software Foundation.
*
*		This program is distributed in the hope that it will be useful,
*		but WITHOUT ANY WARRANTY; without even the implied warranty of
*		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
*		GNU General Public License for more details.
*
*		You should have received a copy of the GNU General Public License
*		along with this program; if not, contact Novell, Inc.
*
*		To contact Novell about this file by physical or electronic mail,
*		you may find current contact information at www.novell.com
*
* $Id: $
*******************************************************************************/

	$buildhostlist = 
		array( 
			array(
				"name" => "win-x86-32-host",
				"description" => "Windows 2003 Server 32-bit",
				"server" => "localhost",
				"username" => "",
				"password" => "",
				"ostype" => "win",
				"osfamily" => "win",
				"hosttype" => "x86",
				"packagearch" => ""
			),
			array(
				"name" => "osx-powerpc-32-host",
				"description" => "Mac OS X (Darwin) powerpc 32-bit",
				"server" => "",
				"username" => "",
				"password" => "",
				"ostype" => "osx",
				"osfamily" => "unix",
				"hosttype" => "powerpc",
				"packagearch" => ""
			),
			array(
				"name" => "ubuntu-dapper-x86-32-host",
				"description" => "Ubuntu 6.06 LTS x86-32 host",
				"server" => "",
				"username" => "",
				"password" => "",
				"ostype" => "linux",
				"osfamily" => "unix",
				"hosttype" => "x86",
				"packagearch" => "i386"
			),
			array(
				"name" => "opensuse10-powerpc-64-host",
				"description" => "OpenSUSE 10 ppc64",
				"server" => "",
				"username" => "",
				"password" => "",
				"ostype" => "linux",
				"osfamily" => "unix",
				"hosttype" => "powerpc",
				"packagearch" => "ppc64"
			),
			array(
				"name" => "solaris10-sparc-64-host",
				"description" => "Solaris 10",
				"server" => "",
				"username" => "",
				"password" => "",
				"ostype" => "solaris",
				"osfamily" => "unix",
				"hosttype" => "sparc",
				"packagearch" => ""
			),
			array(
				"name" => "hpux11-hppa-64-host",
				"description" => "HP-UX 11",
				"server" => "",
				"username" => "",
				"password" => "",
				"ostype" => "hpux",
				"osfamily" => "unix",
				"hosttype" => "hppa",
				"packagearch" => ""
			),
			array(
				"name" => "aix4-powerpc-64-host",
				"description" => "AIX 4.3",
				"server" => "",
				"username" => "",
				"password" => "",
				"ostype" => "aix",
				"osfamily" => "unix",
				"hosttype" => "ppc64",
				"packagearch" => ""
			),
			array(
				"name" => "sles9-x86-64-host",
				"description" => "SLES 9 x86_64",
				"server" => "",
				"username" => "",
				"password" => "",
				"ostype" => "linux",
				"osfamily" => "unix",
				"hosttype" => "x86",
				"packagearch" => "x86_64"
			),
			array(
				"name" => "opensuse10-x86-32-host",
				"description" => "OpenSUSE 10 x86",
				"server" => "",
				"username" => "",
				"password" => "",
				"ostype" => "linux",
				"osfamily" => "unix",
				"hosttype" => "x86",
				"packagearch" => "i586"
			),
			array(
				"name" => "fedoracore4-x86-32-host",
				"description" => "Fedora Core 4 x86",
				"server" => "",
				"username" => "",
				"password" => "",
				"ostype" => "linux",
				"osfamily" => "unix",
				"hosttype" => "x86",
				"packagearch" => "i586"
			)
		);
		
	$projectlist =
		array(
			array(
				"name" => "ftk",
				"productdir" => "ftk-products",
				"majorver" => "1",
				"minorver" => "1",
				"svnbaseurl" => "https://forgesvn1.novell.com/svn/flaim/trunk",
				"svnsubdir" => "ftk",
				"svnrevdirs" => array(
					"tools",
					"ftk"
				),
				"svnrev" => 0,
				"prevrev" => 0,
				"maintainers" => array( 
					"ahodgkinson@novell.com"
				),
			),
			array(
				"name" => "flaim",
				"productdir" => "flaim-products",
				"majorver" => "4",
				"minorver" => "9",
				"svnbaseurl" => "https://forgesvn1.novell.com/svn/flaim/trunk",
				"svnsubdir" => "flaim",
				"svnrevdirs" => array(
					"tools",
					"ftk",
					"flaim"
				),
				"svnrev" => 0,
				"prevrev" => 0,
				"maintainers" => array( 
					"ahodgkinson@novell.com"
				),
			),
			array(
				"name" => "xflaim",
				"productdir" => "xflaim-products",
				"majorver" => "5",
				"minorver" => "1",
				"svnbaseurl" => "https://forgesvn1.novell.com/svn/flaim/trunk",
				"svnsubdir" => "xflaim",
				"svnrevdirs" => array(
					"ftk",
					"tools",
					"xflaim"
				),
				"svnrev" => 0,
				"prevrev" => 0,
				"maintainers" => array( 
					"ahodgkinson@novell.com"
				)
			)
		);
		
	$productlist =
		array(
		
			// FTK
			
			array(
				"name" => "Source packages",
				"project" => "ftk",
				"buildhost" => "sles9-x86-64-host",
				"wordsize" => "0",
				"targets" => array(
					"srcrpm"
				),
				"outputs" => array(
					"srcpackage",
					"srcrpm"
				)
			),
			array(
				"name" => "Windows (x86) 32-bit binary package",
				"project" => "ftk",
				"buildhost" => "win-x86-32-host",
				"wordsize" => "32",
				"targets" => array(
					"32bit test",
					"32bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Netware (x86) 32-bit binary package",
				"project" => "ftk",
				"buildhost" => "win-x86-32-host",
				"targetos" => "netware",
				"wordsize" => "32",
				"targets" => array(
					"32bit nlm bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Linux x86_64 RPMs and binary package",
				"project" => "ftk",
				"buildhost" => "sles9-x86-64-host",
				"wordsize" => "64",
				"targets" => array(
					"test",
					"bindist",
					"rpms"
				),
				"outputs" => array(
					"binpackage",
					"rpm",
					"develrpm"
				)
			),
			array(
				"name" => "Linux x86 (32-bit) RPMs and binary package",
				"project" => "ftk",
				"buildhost" => "opensuse10-x86-32-host",
				"wordsize" => "32",
				"targets" => array(
					"test",
					"bindist",
					"rpms"
				),
				"outputs" => array(
					"binpackage",
					"rpm",
					"develrpm"
				)
			),
//			array(
//				"name" => "Debian x86 Binary package",
//				"project" => "ftk",
//				"buildhost" => "ubuntu-dapper-x86-32-host",
//				"wordsize" => "32",
//				"targets" => array(
//					"test",
//					"debsrc",
//					"debbin"
//				),
//				"outputs" => array(
//					"debbin",
//				)
//			),
			array(
				"name" => "Linux PPC (32-bit) binary package",
				"project" => "ftk",
				"buildhost" => "opensuse10-powerpc-64-host",
				"wordsize" => "32",
				"targets" => array(
					"32bit test",
					"32bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Linux PPC (64-bit) RPMs and binary package",
				"project" => "ftk",
				"buildhost" => "opensuse10-powerpc-64-host",
				"wordsize" => "64",
				"targets" => array(
					"test",
					"bindist",
					"rpms"
				),
				"outputs" => array(
					"binpackage",
					"rpm",
					"develrpm"
				)
			),
			array(
				"name" => "Mac OS X for PPC 32-bit binary package",
				"project" => "ftk",
				"buildhost" => "osx-powerpc-32-host",
				"wordsize" => "32",
				"targets" => array(
					"test",
					"bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Solaris 10 (sparc) 32-bit binary package",
				"project" => "ftk",
				"buildhost" => "solaris10-sparc-64-host",
				"wordsize" => "32",
				"targets" => array(
					"32bit test",
					"32bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Solaris 10 (sparc) 64-bit binary package",
				"project" => "ftk",
				"buildhost" => "solaris10-sparc-64-host",
				"wordsize" => "64",
				"targets" => array(
					"64bit test",
					"64bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			
			// FLAIM

			array(
				"name" => "Source packages",
				"project" => "flaim",
				"buildhost" => "sles9-x86-64-host",
				"wordsize" => "0",
				"targets" => array(
					"srcrpm"
				),
				"outputs" => array(
					"srcpackage",
					"srcrpm"
				)
			),
			array(
				"name" => "Windows (x86) 32-bit binary package",
				"project" => "flaim",
				"buildhost" => "win-x86-32-host",
				"wordsize" => "32",
				"targets" => array(
					"32bit test",
					"32bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Netware (x86) 32-bit binary package",
				"project" => "flaim",
				"buildhost" => "win-x86-32-host",
				"targetos" => "netware",
				"wordsize" => "32",
				"targets" => array(
					"32bit nlm bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Linux x86_64 RPMs and binary package",
				"project" => "flaim",
				"buildhost" => "sles9-x86-64-host",
				"wordsize" => "64",
				"targets" => array(
					"test",
					"bindist",
					"rpms"
				),
				"outputs" => array(
					"binpackage",
					"rpm",
					"develrpm"
				)
			),
			array(
				"name" => "Linux x86 (32-bit) RPMs and binary package",
				"project" => "flaim",
				"buildhost" => "opensuse10-x86-32-host",
				"wordsize" => "32",
				"targets" => array(
					"test",
					"bindist",
					"rpms"
				),
				"outputs" => array(
					"binpackage",
					"rpm",
					"develrpm"
				)
			),
//			array(
//				"name" => "Debian x86 binary package",
//				"project" => "flaim",
//				"buildhost" => "ubuntu-dapper-x86-32-host",
//				"wordsize" => "32",
//				"targets" => array(
//					"test",
//					"debsrc",
//					"debbin"
//				),
//				"outputs" => array(
//					"debbin",
//				)
//			),
			array(
				"name" => "Linux PPC (32-bit) binary package",
				"project" => "flaim",
				"buildhost" => "opensuse10-powerpc-64-host",
				"wordsize" => "32",
				"targets" => array(
					"32bit test",
					"32bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Linux PPC (64-bit) RPMs and binary package",
				"project" => "flaim",
				"buildhost" => "opensuse10-powerpc-64-host",
				"wordsize" => "64",
				"targets" => array(
					"test",
					"bindist",
					"rpms"
				),
				"outputs" => array(
					"binpackage",
					"rpm",
					"develrpm"
				)
			),
			array(
				"name" => "Mac OS X for PPC 32-bit binary package",
				"project" => "flaim",
				"buildhost" => "osx-powerpc-32-host",
				"wordsize" => "32",
				"targets" => array(
					"test",
					"bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Solaris 10 (sparc) 32-bit binary package",
				"project" => "flaim",
				"buildhost" => "solaris10-sparc-64-host",
				"wordsize" => "32",
				"targets" => array(
					"32bit test",
					"32bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Solaris 10 (sparc) 64-bit binary package",
				"project" => "flaim",
				"buildhost" => "solaris10-sparc-64-host",
				"wordsize" => "64",
				"targets" => array(
					"64bit test",
					"64bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			
			// XFLAIM
			
			array(
				"name" => "Source packages",
				"project" => "xflaim",
				"buildhost" => "sles9-x86-64-host",
				"wordsize" => "0",
				"targets" => array(
					"srcrpm"
				),
				"outputs" => array(
					"srcpackage",
					"srcrpm"
				)
			),
			array(
				"name" => "Windows (x86) 32-bit binary package",
				"project" => "xflaim",
				"buildhost" => "win-x86-32-host",
				"wordsize" => "32",
				"targets" => array(
					"32bit test",
					"32bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Netware (x86) 32-bit binary package",
				"project" => "xflaim",
				"buildhost" => "win-x86-32-host",
				"targetos" => "netware",
				"wordsize" => "32",
				"targets" => array(
					"32bit nlm bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Linux x86_64 RPMs and binary package",
				"project" => "xflaim",
				"buildhost" => "sles9-x86-64-host",
				"wordsize" => "64",
				"targets" => array(
					"test",
					"bindist",
					"rpms"
				),
				"outputs" => array(
					"binpackage",
					"rpm",
					"develrpm"
				)
			),
			array(
				"name" => "Linux x86 (32-bit) RPMs and binary package",
				"project" => "xflaim",
				"buildhost" => "opensuse10-x86-32-host",
				"wordsize" => "32",
				"targets" => array(
					"test",
					"bindist",
					"rpms"
				),
				"outputs" => array(
					"binpackage",
					"rpm",
					"develrpm"
				)
			),
//			array(
//				"name" => "Debian x86 binary package",
//				"project" => "xflaim",
//				"buildhost" => "ubuntu-dapper-x86-32-host",
//				"wordsize" => "32",
//				"targets" => array(
//					"test",
//					"debsrc",
//					"debbin"
//				),
//				"outputs" => array(
//					"debbin",
//				)
//			),
			array(
				"name" => "Linux PPC (32-bit) binary package",
				"project" => "xflaim",
				"buildhost" => "opensuse10-powerpc-64-host",
				"wordsize" => "32",
				"targets" => array(
					"32bit test",
					"32bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Linux PPC (64-bit) RPMs and binary package",
				"project" => "xflaim",
				"buildhost" => "opensuse10-powerpc-64-host",
				"wordsize" => "64",
				"targets" => array(
					"test",
					"bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Linux PPC (64-bit) RPMs and binary package",
				"project" => "xflaim",
				"buildhost" => "opensuse10-powerpc-64-host",
				"wordsize" => "64",
				"targets" => array(
					"test",
					"bindist",
					"rpms"
				),
				"outputs" => array(
					"binpackage",
					"rpm",
					"develrpm"
				)
			),
			array(
				"name" => "Mac OS X for PPC 32-bit binary package",
				"project" => "xflaim",
				"buildhost" => "osx-powerpc-32-host",
				"wordsize" => "32",
				"targets" => array(
					"test",
					"bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Solaris 10 (sparc) 32-bit binary package",
				"project" => "xflaim",
				"buildhost" => "solaris10-sparc-64-host",
				"wordsize" => "32",
				"targets" => array(
					"32bit test",
					"32bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			),
			array(
				"name" => "Solaris 10 (sparc) 64-bit binary package",
				"project" => "xflaim",
				"buildhost" => "solaris10-sparc-64-host",
				"wordsize" => "64",
				"targets" => array(
					"64bit test",
					"64bit bindist"
				),
				"outputs" => array(
					"binpackage"
				)
			)
		);
		
	$buildtestlist = array();
	
?>
