<?php
/*******************************************************************************
* Desc:	PHP script for remote FLAIM builds
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

include "buildcfg.php";

/*******************************************************************************
Desc:
*******************************************************************************/
function rexecCommand( $connection, $command, &$output, $logfhdl = NULL)
{
	$output = NULL;
	$exitstatus = -1;
	
	$logline = "$command\n"; 
	printf( "%s", $logline);
	
	if( $logfhdl != NULL)
	{
		fprintf( $logfhdl, "%s", $logline);
	}
	
	$command .= "; echo \"EXITSTATUS = \$?\"";
	if( !($stdio = ssh2_exec( $connection, $command)))
	{
		die( "Unable to execute command.");
	}
	
	$stderr = ssh2_fetch_stream( $stdio, SSH2_STREAM_STDERR);
	stream_set_blocking( $stdio, true);

	while( $line = fgets( $stdio))
	{
		if( strpos( $line, "EXITSTATUS = ") === 0)
		{
			$exitstatus = (int)$line[ 13];
			break;
		}
		
		$output .= $line;
		$logline = "(stdout) $line";
		printf( "%s", $logline);
		
		if( $logfhdl != NULL)
		{
			fprintf( $logfhdl, "%s", $logline);
		}

		while( $line = fgets( $stderr))
		{
			$logline = "(stderr) $line";
			printf( "%s", $logline);
			
			if( $logfhdl != NULL)
			{
				fprintf( $logfhdl, "%s", $logline);
			}
		}
		
		flush();
		if( $logfhdl != NULL)
		{
			fflush( $logfhdl);
		}
	}
	
	while( $line = fgets( $stderr))
	{
		$logline = "(stderr) $line";
		printf( "%s", $logline);

		if( $logfhdl != NULL)
		{
			fprintf( $logfhdl, "%s", $logline);
		}
	}
	
	flush();
	if( $logfhdl != NULL)
	{
		fflush( $logfhdl);
	}
	
	$output = trim( $output);
	fclose( $stderr);
	fclose( $stdio);
	
	return( $exitstatus);
}	

/*******************************************************************************
Desc:
*******************************************************************************/
function lexecCommand( $command, &$output, $logfhdl = NULL)
{
	$exitstatus = -1;
	$output = NULL;
	
	$descriptorspec = array(
		0 => array( "pipe", "r"),  // stdin
		1 => array( "pipe", "w"),  // stdout
		2 => array( "pipe", "w")	// stderr
	);
	
	$process = proc_open( "$command", 
		$descriptorspec, $pipes, NULL, NULL);
	
	if( is_resource( $process))
	{
		fclose( $pipes[ 0]);
		
		while( !feof( $pipes[ 1]))
		{
			$line = fgets( $pipes[ 1]);
			echo $line;
			$output .= $line;
		}

		while( !feof( $pipes[ 2]))		
		{
			$line = fgets( $pipes[ 2]);
			echo $line;
			$output .= $line;
		}
		
		fclose( $pipes[ 2]);
		fclose( $pipes[ 1]);
	
		$exitstatus = proc_close( $process);
	}
	
	return( $exitstatus);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function sshConnectWithPassword( $server, $username, $password)
{
	$failed = false;
	
	try
	{
		printf( "Verifying that %s supports password authentication ... ", $server);
		
		if( !($connection = ssh2_connect( $server, 22, null, $callbacks)))
		{
			throw new Exception( "Could not connect.");
		}
		
		$auth_methods = ssh2_auth_none( $connection, "nobody");
		
		if( !in_array( "password", $auth_methods))
		{
			printf( "it doesn't.\n");
			throw new Exception( "Could not connect.");
		}
		else
		{
			printf( "it does.\n");
		}
		
		unset( $connection);
		
		printf( "Connecting to server via ssh ... ");
		
		if( !($connection = ssh2_connect( $server, 22, null, $callbacks)))
		{
			throw new Exception( "Could not connect.");
		}
		
		if (ssh2_auth_password( $connection, $username, $password))
		{
			printf( "successful.\n");
		} 
		else
		{
			throw new Exception( "Could not connect.");
		}
	}
	catch (Exception $e)
	{
		return( NULL);
	}
	
	return( $connection);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function sshConnectWithPPK( $server, $username, $pubkeyfile, $privkeyfile)
{
	try
	{
		printf( "Connecting to server via ssh ... ");
		
		if( !($connection = ssh2_connect( $server, 22)))
		{
			throw new Exception( "Could not connect.");
		}
		
		if (ssh2_auth_pubkey_file( $connection, $username, 
			$pubkeyfile, $privkeyfile))
		{
			printf( "successful.\n");
		} 
		else
		{
			throw new Exception( "Could not connect.");
		}
	}
	catch (Exception $e)
	{
		return( NULL);
	}
	
	return( $connection);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function findRemoteMakeUtil( $connection)
{
	rexecCommand( $connection, "make --version", $tmp);
	
	if( stripos( $tmp, "GNU Make") !== false)
	{
		rexecCommand( $connection, "which make", $tmp);
		return( $tmp);
	}
	
	rexecCommand( $connection, "gmake --version", $tmp);
	
	if( stripos( $tmp, "GNU Make") !== false)
	{
		rexecCommand( $connection, "which gmake", $tmp);
		return( $tmp);
	}
	
	return( NULL);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function remoteMakeTempDir( $connection, $ostype, &$tmpname)
{
	$exitstatus = 0;
	$tmpname = NULL;
	
	try
	{
		if( stripos( $ostype, "linux") !== false ||
			 stripos( $ostype, "solaris") !== false)
		{
			if( ($exitstatus = rexecCommand( $connection, 
				"mktemp -d", $tmpname)) != 0)
			{
				throw new Exception( "Error creating directory: $exitstatus");
			}
		}
		else if( stripos( $ostype, "hpux") !== false)
		{
			if( ($exitstatus = rexecCommand( $connection, "mktemp", $tmpname)) != 0)
			{
				throw new Exception( "Error creating directory: $exitstatus");
			}
			
			if( ($exitstatus = rexecCommand( $connection, "mkdir $tmpname")) != 0)
			{
				throw new Exception( "Error creating directory: $exitstatus");
			}
		}
		else if( stripos( $ostype, "osx") !== false)
		{
			if( ($exitstatus = rexecCommand( $connection, 
				"mktemp -t tmp -d tmp.XXXXXX | grep /tmp", $tmpname)) != 0)
			{
				throw new Exception( "Error creating directory: $exitstatus");
			}
		}
		else
		{
			$exitstatus = -1;
			throw new Exception( "Don't know how to make a temp dir on this host.");
		}
	}
	catch (Exception $e)
	{
	}
	
	return( $exitstatus);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function localMakeTempDir( $ostype, &$tmpname)
{
	$exitstatus = 0;
	$tmpname = NULL;
	
	try
	{
		if( ($tmpname = tempnam( ".", "tmp")) === false)
		{
			$exitstatus = -1;
			throw new Exception( "Error creating directory: $exitstatus");
		}
		
		unlink( $tmpname);
		mkdir( $tmpname);
	}
	catch (Exception $e)
	{
	}
	
	return( $exitstatus);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function remoteRemoveDir( $connection, $ostype, $dir)
{
	try
	{
		if( stripos( $ostype, "linux") !== false ||
			 stripos( $ostype, "solaris") !== false ||
			 stripos( $ostype, "osx") !== false ||
			 stripos( $ostype, "hpux") !== false)
		{
			rexecCommand( $connection, "rm -rf $dir", $tmp);
			return( $tmp);
		}
		else
		{
			throw new Exception( "Don't know how to remove a dir on this host.");
		}
	}
	catch (Exception $e)
	{
	}
	
	return( NULL);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function localRemoveDir( $dir)
{
	$dir_contents = scandir( $dir);
	
	foreach( $dir_contents as $item) 
	{
		$fullpath = $dir.DIRECTORY_SEPARATOR.$item;
		
		if( is_dir( $fullpath) && $item != '.' && $item != '..') 
		{
			localRemoveDir( $fullpath);
		}
		else if( file_exists( $fullpath) && $item != '.' && $item != '..') 
		{
			chmod( $fullpath, 0777);
			unlink( $fullpath);
		}
	}
	
	chmod( $dir, 0777);
	rmdir( $dir);
}
	
/*******************************************************************************
Desc:
*******************************************************************************/
function remoteSubversionCheckout( 
	$connection, 
	$osfamily, 
	$hosttype, 
	$tmpdir, 
	$svnurl, 
	$svnrev)
{
	if( stripos( $osfamily, "unix") !== false)
	{
		$cmd = "source ~/.profile;";
		$cmd .= " svn checkout --non-interactive -r $svnrev $svnurl $tmpdir"; 
	}
	
	rexecCommand( $connection, $cmd, $tmp);
							
	if( stripos( $tmp, "certificate verification failed") !== false)
	{
		printf( "Server certificate is not recognized by Subversion.\n");
		return( false);		
	}
							
	if( strstr( $tmp, "Checked out revision $svnrev.") === false)
	{
		printf( "Checkout failed.\n");
		return( false);
	}
	
	return( true);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function localSubversionCheckout( 
	$osfamily, 
	$hosttype, 
	$tmpdir, 
	$svnurl, 
	$svnrev)
{
	$cmd = "svn checkout --non-interactive -r $svnrev $svnurl $tmpdir";
	system( $cmd, $iRetVal);
	
	if( iRetVal != 0)
	{
		printf( "Checkout failed.\n");
		return( false);
	}
	
	return( true);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function remoteBuild( 
	$connection,
	$osfamily,
	$hosttype,
	$blddir,
	$target,
	&$stdout,
	$logfhdl)
{
	if( stripos( $osfamily, "unix") !== false)
	{
		$makecmd = findRemoteMakeUtil( $connection);
		$bldcmd = "source ~/.profile; cd $blddir; $makecmd $target";
	}
	
	return( rexecCommand( $connection, $bldcmd, $stdout, $logfhdl));
}

/*******************************************************************************
Desc:
*******************************************************************************/
function localBuild( 
	$osfamily,
	$hosttype,
	$blddir,
	$target,
	&$stdout,
	$logfhdl)
{
	if( stripos( $osfamily, "win") !== false)
	{
		$bldcmd = "cd $blddir && make $target";
	}
	else
	{
		throw new Exception( "Don't know how to do a build on this host.");
	}
	
	printf( "$bldcmd\n");
	return( lexecCommand( $bldcmd, $stdout, $logfhdl));
}

/*******************************************************************************
Desc:
*******************************************************************************/
function svnCalcHighRev( $svnurl, $svnsubdir)
{
	$highrev = 0;
	
	if( $svnsubdir)
	{
		$svnurl .= '/' . $svnsubdir;
	}
	
	exec( "svn info --non-interactive -R $svnurl", $cmdoutput);
	
	foreach( $cmdoutput as $line)
	{
		if( stripos( $line, "Last Changed Rev: ") !== false)
		{
			$line = trim( $line);
			$rev = substr( $line, stripos( $line, ": ") + 2);
			if( (int)$rev > (int)$highrev)
			{
				$highrev = $rev;
			}
		}
	}
	
	return( $highrev);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function emailFile( $subject, $toaddr, $filename)
{
	$cmd  = "blat.exe";
	$cmd .= " $filename -to $toaddr -server mail.myserver.com";
	$cmd .= " -subject \"$subject\" -f flaimbuild@myserver.com"; 
	
	printf( "e-mail command: %s\n", $cmd);
	return( trim( exec( $cmd))); 
}

/*******************************************************************************
Desc:
*******************************************************************************/
function getRPMName( $project, $buildhost, $svnrev)
{
	$rpmname  = "lib";
	$rpmname .= $project[ "name"];
	$rpmname .= "-";
	$rpmname .= $project[ "majorver"];
	$rpmname .= ".";
	$rpmname .= $project[ "minorver"];
	$rpmname .= ".";
	$rpmname .= $svnrev;
	$rpmname .= "-1";
	$rpmname .= ".";
	$rpmname .= $buildhost[ "packagearch"];
	$rpmname .= ".rpm";
	
	return( $rpmname);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function getDevelRPMName( $project, $buildhost, $svnrev)
{
	$rpmname  = "lib";
	$rpmname .= $project[ "name"];
	$rpmname .= "-devel-";
	$rpmname .= $project[ "majorver"];
	$rpmname .= ".";
	$rpmname .= $project[ "minorver"];
	$rpmname .= ".";
	$rpmname .= $svnrev;
	$rpmname .= "-1";
	$rpmname .= ".";
	$rpmname .= $buildhost[ "packagearch"];
	$rpmname .= ".rpm";
	
	return( $rpmname);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function getSrcRPMName( $project, $buildhost, $svnrev)
{
	$rpmname  = "lib";
	$rpmname .= $project[ "name"];
	$rpmname .= "-";
	$rpmname .= $project[ "majorver"];
	$rpmname .= ".";
	$rpmname .= $project[ "minorver"];
	$rpmname .= ".";
	$rpmname .= $svnrev;
	$rpmname .= "-1";
	$rpmname .= ".src.rpm";
	
	return( $rpmname);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function getBinPackageName( $product, $project, $buildhost, $svnrev)
{
	$rpmname  = "lib";
	$rpmname .= $project[ "name"];
	$rpmname .= "-";
	$rpmname .= $project[ "majorver"];
	$rpmname .= ".";
	$rpmname .= $project[ "minorver"];
	$rpmname .= ".";
	$rpmname .= $svnrev;
	$rpmname .= "-";
	
	if( isset( $product[ "targetos"]))
	{
		$rpmname .= $product[ "targetos"];
	}
	else
	{
		$rpmname .= $buildhost[ "ostype"];
	}
	
	$rpmname .= "-";
	$rpmname .= $buildhost[ "hosttype"];
	$rpmname .= "-";
	$rpmname .= $product[ "wordsize"];
	$rpmname .= "-bin.tar.gz";
	
	return( $rpmname);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function getSrcPackageName( $project, $svnrev)
{
	$rpmname  = "lib";
	$rpmname .= $project[ "name"];
	$rpmname .= "-";
	$rpmname .= $project[ "majorver"];
	$rpmname .= ".";
	$rpmname .= $project[ "minorver"];
	$rpmname .= ".";
	$rpmname .= $svnrev;
	$rpmname .= ".tar.gz";
	
	return( $rpmname);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function getDebianBinaryPackageName( $project, $buildhost, $svnrev)
{
	$pkgname  = "lib";
	$pkgname .= $project[ "name"];
	$pkgname .= "_";
	$pkgname .= $project[ "majorver"];
	$pkgname .= ".";
	$pkgname .= $project[ "minorver"];
	$pkgname .= ".";
	$pkgname .= $svnrev;
	$pkgname .= "-";
	$pkgname .= $buildhost[ "packagearch"];
	$pkgname .= ".deb";
	
	return( $pkgname);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function parseSrcPackageName( 
	$packagename, 
	&$ostype, 
	&$hosttype,
	&$packagearch,
	&$projectname, 
	&$svnrev, 
	&$wordsize)
{
	$ostype = "";
	$hosttype = "";
	$packagearch = "";
	$wordsize = 0;
	
	if( strncmp( $packagename, "lib", 3) != 0)
	{
		return( false);
	}
	
	$packagename = substr( $packagename, 3);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$projectname = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$majorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$minorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$svnrev = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( strncmp( $packagename, "tar.gz", 10) != 0)
	{
		return( false);
	}
	
	return( true);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function parseBinPackageName( 
	$packagename, 
	&$ostype, 
	&$hosttype,
	&$packagearch,
	&$projectname, 
	&$svnrev, 
	&$wordsize)
{
	$packagearch = "";
	
	if( strncmp( $packagename, "lib", 3) != 0)
	{
		return( false);
	}
	
	$packagename = substr( $packagename, 3);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$projectname = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$majorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$minorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$svnrev = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$ostype = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$hosttype = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$wordsize = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( strncmp( $packagename, "bin.tar.gz", 10) != 0)
	{
		return( false);
	}
	
	return( true);
}
								
/*******************************************************************************
Desc:
*******************************************************************************/
function parseRPMName( 
	$packagename, 
	&$ostype, 
	&$hosttype,
	&$packagearch,
	&$projectname, 
	&$svnrev, 
	&$wordsize)
{
	$ostype = "linux";
	$hosttype = "";
	
	if( strncmp( $packagename, "lib", 3) != 0)
	{
		return( false);
	}
	
	$packagename = substr( $packagename, 3);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$projectname = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$majorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$minorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$svnrev = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$packagerev = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$packagearch = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( strncmp( $packagename, "rpm", 3) != 0)
	{
		return( false);
	}
	
	return( true);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function parseDEBName( 
	$packagename, 
	&$ostype, 
	&$hosttype,
	&$packagearch,
	&$projectname, 
	&$svnrev, 
	&$wordsize)
{
	$ostype = "linux";
	$hosttype = "";
	
	if( strncmp( $packagename, "lib", 3) != 0)
	{
		return( false);
	}
	
	$packagename = substr( $packagename, 3);
	
	if( ($strOffset = strpos( $packagename, "_")) === false)
	{
		return( false);
	}
	
	$projectname = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$majorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$minorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$svnrev = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$packagearch = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( strncmp( $packagename, "deb", 3) != 0)
	{
		return( false);
	}
	
	return( true);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function parseDevelRPMName( 
	$packagename, 
	&$ostype, 
	&$hosttype,
	&$packagearch,
	&$projectname, 
	&$svnrev, 
	&$wordsize)
{
	$ostype = "linux";
	$hosttype = "";
	
	if( strncmp( $packagename, "lib", 3) != 0)
	{
		return( false);
	}
	
	$packagename = substr( $packagename, 3);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$projectname = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( strncmp( $packagename, "devel-", 6) != 0)
	{
		return( false);
	}
	
	$packagename = substr( $packagename, 6);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$majorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$minorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$svnrev = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$packagerev = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$packagearch = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( strncmp( $packagename, "rpm", 3) != 0)
	{
		return( false);
	}
	
	return( true);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function parseSrcRPMName( 
	$packagename, 
	&$ostype, 
	&$hosttype,
	&$packagearch,
	&$projectname, 
	&$svnrev, 
	&$wordsize)
{
	$ostype = "linux";
	$hosttype = "";
	$packagearch = "";
	$worsize = 0;
	
	if( strncmp( $packagename, "lib", 3) != 0)
	{
		return( false);
	}
	
	$packagename = substr( $packagename, 3);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$projectname = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$majorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$minorver = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, "-")) === false)
	{
		return( false);
	}
	
	$svnrev = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( ($strOffset = strpos( $packagename, ".")) === false)
	{
		return( false);
	}
	
	$packagerev = substr( $packagename, 0, $strOffset);
	$packagename = substr( $packagename, $strOffset + 1);
	
	if( strncmp( $packagename, "src.rpm", 7) != 0)
	{
		return( false);
	}
	
	return( true);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function parsePackageName(
	$packagename,
	&$packagetype,
	&$ostype, 
	&$hosttype,
	&$packagearch,
	&$projectname, 
	&$svnrev, 
	&$wordsize)
{
	if( parseBinPackageName( $packagename, $ostype, 
		$hosttype, $packagearch, $projectname, $svnrev, $wordsize)) 
	{
		$packagetype = "binpackage";
		return( true);
	}
	else if( parseSrcPackageName( $packagename, $ostype, 
		$hosttype, $packagearch, $projectname, $svnrev, $wordsize)) 
	{
		$packagetype = "srcpackage";
		return( true);
	}
	else if( parseDevelRPMName( $packagename, $ostype, 
		$hosttype, $packagearch, $projectname, $svnrev, $wordsize))
	{
		$packagetype = "develrpm";
		return( true);
	}
	else if( parseSrcRPMName( $packagename, $ostype, 
		$hosttype, $packagearch, $projectname, $svnrev, $wordsize))
	{
		$packagetype = "srcrpm";
		return( true);
	}
	else if( parseRPMName( $packagename, $ostype, 
		$hosttype, $packagearch, $projectname, $svnrev, $wordsize))
	{
		$packagetype = "rpm";
		return( true);
	}
	else if( parseDEBName( $packagename, $ostype,
		$hosttype, $packagearch, $projectname, $svnrev, $wordsize))
	{
		$packagetype = "deb";
		return( true);
	}
	
	return( false);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function translateRPMArch( $rpmarch, &$hosttype, &$wordsize)
{
	if( $rpmarch == "i386" || $rpmarch == "i486" || 
		 $rpmarch == "i586" || $rpmarch == "i686")
	{
		$hosttype = "x86";
		$wordsize = "32";
		return( true);
	}
	else if( $rpmarch == "x86_64")
	{
		$hosttype = "x86";
		$wordsize = "64";
		return( true);
	}
	else if( $rpmarch == "ppc")
	{
		$hosttype = "powerpc";
		$wordsize = "32";
		return( true);
	}
	else if( $rpmarch == "ppc64")
	{
		$hosttype = "powerpc";
		$wordsize = "64";
		return( true);
	}
	
	return( false);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function translateDEBArch( $debarch, &$hosttype, &$wordsize)
{
	if( $debarch == "i386")
	{
		$hosttype = "x86";
		$wordsize = "32";
		return( true);
	}
	else if( $debarch == "amd64")
	{
		$hosttype = "x86";
		$wordsize = "64";
		return( true);
	}
	else if( $debarch == "sparc")
	{
		$hosttype = "sparc";
		$wordsize = "32";
		return( true);
	}
	else if( $debarch == "powerpc")
	{
		$hosttype = "powerpc";
		$wordsize = "64";
		return( true);
	}
	
	return( false);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function findFile( $dirlist, $filename, &$retpath)
{
	foreach( $dirlist as $item)
	{
		if( $item[ "name"] == $filename)
		{
			$retpath = $item[ "path"];
			break;
		}
	}
	
	return( $retpath == null ? false : true);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function buildProduct( $product)
{
	global $buildhostlist;
	global $projectlist;
	
	foreach( $projectlist as $project)
	{
		if( $project[ "name"] == $product[ "project"])
		{
			break;
		}
	}
	
	if( $project[ "name"] != $product[ "project"])
	{
		printf( "Unable to locate project '%s' in project list.\n", 
			$product[ "project"]);
		return( false);
	}
	
	foreach( $buildhostlist as $buildhost)
	{
		if( $buildhost[ "name"] == $product[ "buildhost"])
		{
			break;
		}
	}
	
	if( $buildhost[ "name"] != $product[ "buildhost"])
	{
		printf( "Unable to locate build host '%s' in host list.\n", 
			$product[ "buildhost"]);
		return( false);
	}
	
	$projname = $project[ "name"];
	$productdir = $project[ "productdir"];
	$svnbaseurl = $project[ "svnbaseurl"];
	$svnsubdir = $project[ "svnsubdir"];
	$svnrev = $project[ "svnrev"];
	
	// Make sure the local product directory exists
	
	if( !file_exists( $productdir))
	{
		mkdir( $productdir);
	}
	
	$buildFailed = false;
	$server = $buildhost[ "server"];
	$username = $buildhost[ "username"];
	$password = $buildhost[ "password"];
	$logfilename = $product[ "project"] . ".log";
	
	if( stripos( $server, "localhost") !== FALSE)
	{
		$localhost = true;
	}
	else
	{
		$localhost = false;
	}
	
	unset( $tmpdir);
	
	if( ($logfhdl = fopen( $logfilename, "w+")) === false)
	{
		throw new Exception( "Unable to open log file $logfilename");
	}
	
	$bldhost_ostype = $buildhost[ "ostype"];
	$bldhost_osfamily = $buildhost[ "osfamily"];
	$bldhost_hosttype = $buildhost[ "hosttype"];
		
	printf( "Build host operating system = %s\n", $bldhost_ostype);
	printf( "Build host operating system family = %s\n", $bldhost_osfamily);
	printf( "Build host type = %s\n", $bldhost_hosttype);
		
	$prodpaths[ "base"][ "srcpackage"] = getSrcPackageName( $project, $svnrev);
	$prodpaths[ "base"][ "binpackage"] = getBinPackageName( $product, $project, $buildhost, $svnrev);
	$prodpaths[ "base"][ "rpm"] = getRPMName( $project, $buildhost, $svnrev);
	$prodpaths[ "base"][ "srcrpm"] = getSrcRPMName( $project, $buildhost, $svnrev);
	$prodpaths[ "base"][ "develrpm"] = getDevelRPMName( $project, $buildhost, $svnrev);
	$prodpaths[ "base"][ "debbin"] = getDebianBinaryPackageName( $project, $buildhost, $svnrev);
	
	$prodpaths[ "local"][ "srcpackage"] = $productdir.DIRECTORY_SEPARATOR.getSrcPackageName( $project, $svnrev);
	$prodpaths[ "local"][ "binpackage"] = $productdir.DIRECTORY_SEPARATOR.getBinPackageName( $product, $project, $buildhost, $svnrev);
	$prodpaths[ "local"][ "rpm"] = $productdir.DIRECTORY_SEPARATOR.getRPMName( $project, $buildhost, $svnrev);
	$prodpaths[ "local"][ "srcrpm"] = $productdir.DIRECTORY_SEPARATOR.getSrcRPMName( $project, $buildhost, $svnrev);
	$prodpaths[ "local"][ "develrpm"] = $productdir.DIRECTORY_SEPARATOR.getDevelRPMName( $project, $buildhost, $svnrev);
	$prodpaths[ "local"][ "debbin"] = $productdir.DIRECTORY_SEPARATOR.getDebianBinaryPackageName( $project, $buildhost, $svnrev);

	try
	{
		// See if we already have already built the deliverable and
		// have a copies locally
		
		$buildNeeded = false;
		
		foreach( $product[ "outputs"] as $prodout)
		{
			$localprod = $prodpaths[ "local"][ $prodout];
			printf( "Looking for ($prodout) $localprod ... ");
		
			if( file_exists( $localprod) == false)
			{
				printf( "not found.  Build needed.\n");
				$buildNeeded = true;
				break;				
			}
			else
			{
				printf( "found.\n");
			}
		}
		
		if( $buildNeeded == true)
		{
			if( $localhost == false)
			{
				if( ($connection = sshConnectWithPassword( $server, $username,
						$password)) == NULL)
				{
					throw new Exception( "Unable to connect to remote host.");
				}
				
				if( remoteMakeTempDir( $connection, $bldhost_ostype, $tmpdir) != 0)
				{
					throw new Exception( "Could not create temporary directory.");
				}
			}
			else
			{
				if( localMakeTempDir( $bldhost_ostype, $tmpdir) != 0)
				{
					throw new Exception( "Could not create temporary directory.");
				}
			}
			
			printf( "Created %s.\n", $tmpdir);
		
			// Checkout the project

			if( $localhost == false)
			{			
				if( !remoteSubversionCheckout( $connection, $bldhost_osfamily, 
						$bldhost_hosttype, $tmpdir, $svnbaseurl, $svnrev))
				{
					throw new Exception( "Unable to checkout project");
				}
			}
			else
			{
				if( !localSubversionCheckout( $bldhost_osfamily, 
						$bldhost_hosttype, $tmpdir, $svnbaseurl, $svnrev))
				{
					throw new Exception( "Unable to checkout project");
				}
			}
		
			// Build each target
			
			foreach( $product[ "targets"] as $target)
			{
				if( $localhost == false)
				{
					if( remoteBuild( $connection, $bldhost_osfamily, 
						$bldhost_hosttype, $tmpdir."/".$svnsubdir, 
						$target, $stdout, $logfhdl) != 0)
					{
						throw new Exception( "Build failed");
					}
				}
				else
				{
					if( localBuild( $bldhost_osfamily, 
						$bldhost_hosttype, $tmpdir.DIRECTORY_SEPARATOR.$svnsubdir, 
						$target, $stdout, $logfhdl) != 0)
					{
						throw new Exception( "Build failed");
					}
				}
			}
			
			if( $localhost == false)
			{
				$remotedirlist = getRemoteDirectoryListing( $connection, 
											$tmpdir."/".$svnsubdir);
			}
			else
			{
				$remotedirlist = getDirectoryListing( 
					$tmpdir.DIRECTORY_SEPARATOR.$svnsubdir);
			}
				
			// Copy the deliverables to the product directory
			
			printf( "Copying product deliverables ...\n");
			
			foreach( $product[ "outputs"] as $prodout)
			{
				// Find the file
				
				if( !findFile( $remotedirlist, $prodpaths[ "base"][ $prodout], 
					$prodpaths[ "remote"][ $prodout]))
				{
					printf( "Unable to find %s.  Build failed.\n",
						$prodpaths[ "base"][ $prodout]);
						
					throw new Exception( "Unable to find file.  Build failed.");
				}
				
				if( $localhost == false)
				{
					printf( ">  (local) %s\n", $prodpaths[ "local"][ $prodout]);
					printf( ">  (remote) %s\n", $prodpaths[ "remote"][ $prodout]);
					
					if( !ssh2_scp_recv( $connection, $prodpaths[ "remote"][ $prodout],
						$prodpaths[ "local"][ $prodout]))
					{
						throw new Exception( "Unable to transfer file.  Build failed.");
					}
				}
				else
				{
					printf( ">  (local) %s\n", $prodpaths[ "local"][ $prodout]);
					printf( ">  (remote) %s\n", $prodpaths[ "remote"][ $prodout]);
					
					if( !copy( $prodpaths[ "remote"][ $prodout], 
								  $prodpaths[ "local"][ $prodout]))
					{
						throw new Exception( "Unable to transfer file.  Build failed.");
					}
				}
			}
		}
		
		printf( "done.\n");
	}
	catch (Exception $e)
	{
		printf( "Caught exception: %s\n", $e->getMessage());
		$buildFailed = true;
	}
	
	if( isset( $tmpdir))
	{
		printf( "Removing $tmpdir ... ");
		
		if( $localhost == false)
		{
			remoteRemoveDir( $connection, $bldhost_ostype, $tmpdir);
		}
		else
		{
			localRemoveDir( $tmpdir);
		}
		
		printf( "done.\n");
	}
	
	fclose( $logfhdl);
	
	if( $buildFailed == true)
	{
		foreach( $project[ "maintainers"] as $maintainer)
		{
			printf( "Sending build log to $maintainer ... ");
			emailFile( $project[ "name"]." build failed.", 
					$maintainer, $logfilename);
			printf( "done.\n");
		}
	}
	
	unlink( $logfilename);
	unset( $connection);
	
	return( $buildFailed == true ? false : true);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function getDirectoryListing( $path)
{
   $files = array();
   $i = 0;
   
	if( is_dir( $path))
	{
		if( $dh = opendir( $path))
		{
			while( ($file = readdir( $dh)) !== false)
			{
				if ($file == "." || $file == "..")
				{
					continue;
				}
				
				$fullpath = $path . "/" . $file;
				$fileStat = stat( $fullpath);
				
				$files[ $i][ "name"] = $file;
				$files[ $i][ "type"] = filetype( $fullpath);
				$files[ $i][ "path"] = $fullpath; 
				$files[ $i][ "size"] = $fileStat[ "size"];
				$files[ $i][ "mtime"] = $fileStat[ "mtime"];
				$i++;
				
				if( $files[ $i - 1][ "type"] == "dir")
				{
					$subdirList = getDirectoryListing( $fullpath);
					foreach( $subdirList as $subdirFile)
					{
						$files[ $i++] = $subdirFile;
					}
				}
			}
				
			closedir( $dh);
		}
	}
		
	return( $files);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function getRemoteDirectoryListing( $connection, $path)
{
	$sftp = ssh2_sftp( $connection);
	$ftpPrefix = "ssh2.sftp://$sftp";
	$remotedirpath = $ftpPrefix.$path;
	
	printf( "Building remote file list ... ");
	$remotedirlist = getDirectoryListing( $remotedirpath);
	printf( "done.\n");
	
	foreach( $remotedirlist as &$remoteitem)
	{
		$remoteitem[ "path"] = substr( 
					$remoteitem[ "path"], strlen( $ftpPrefix));
	}
	
	unset( $sftp);
	return( $remotedirlist);
}

/*******************************************************************************
Desc:
*******************************************************************************/
function updateFTPServer( $localproductdir)
{
	$server = "ftp.myserver.com";
	$username = "myusername";
	$pubkeyfile = "publickey";
	$privkeyfile = "privatekey";
	$updateCount = 0;

	if( ($connection = sshConnectWithPPK( $server, $username,
			$pubkeyfile, $privkeyfile)) == NULL)
	{
		throw new Exception( "Unable to connect to remote host.");
	}
	
	$sftp = ssh2_sftp( $connection);
	
	printf( "Building remote file list ... ");
	$remotedirpath = "ssh2.sftp://$sftp/flaim/development";
	$remotedirlist = getDirectoryListing( $remotedirpath);
	printf( "done.\n");
	
	$rprodcount = 0;
	$rprodlist = array();
	
	foreach( $remotedirlist as $remoteitem)
	{
		if( $remoteitem[ "type"] == "file")
		{
			if( !parsePackageName( $remoteitem[ "name"], $rptype, $rpostype, 
				$rphosttype, $rparch, $rpprojectname, $rpsvnrev, $rpwordsize)) 
			{
				continue;
			}
			
			$rprodlist[ $rprodcount][ "type"] = $rptype;
			$rprodlist[ $rprodcount][ "project"] = $rpprojectname;
			$rprodlist[ $rprodcount][ "ostype"] = $rpostype;
			$rprodlist[ $rprodcount][ "hosttype"] = $rphosttype;
			$rprodlist[ $rprodcount][ "packagearch"] = $rparch;
			$rprodlist[ $rprodcount][ "wordsize"] = $rpwordsize;
			$rprodlist[ $rprodcount][ "svnrev"] = $rpsvnrev;
			$rprodlist[ $rprodcount][ "timestamp"] = $remoteitem[ "mtime"];
			$rprodlist[ $rprodcount][ "path"] = $remoteitem[ "path"];
			$rprodlist[ $rprodcount][ "name"] = $remoteitem[ "name"];
			$rprodcount++;
		}
	}
	
	printf( "Building local file list of %s ... ", $localproductdir);
	$localdirlist = getDirectoryListing( $localproductdir);
	printf( "done.\n");
	
	$lprodcount = 0;
	$lprodlist = array();
	
	foreach( $localdirlist as $localitem)
	{
		if( $localitem[ "type"] == "file")
		{
			if( !parsePackageName( $localitem[ "name"], $lptype, $lpostype, 
				$lphosttype, $lparch, $lpprojectname, $lpsvnrev, $lpwordsize)) 
			{
				printf( "Parse failed on %s\n", $localitem[ "name"]);
				continue;
			}
			
			// Keep only the most current versions of each local file in the list
		
			$addItemToList = true;
			foreach( $lprodlist as &$tmpitem)
			{
				if( $tmpitem[ "type"] == $lptype &&
					 $tmpitem[ "project"] == $lpprojectname &&
					 $tmpitem[ "ostype"] == $lpostype &&
					 $tmpitem[ "hosttype"] == $lphosttype &&
					 $tmpitem[ "packagearch"] == $lparch &&
					 $tmpitem[ "wordsize"] == $lpwordsize)
				{
					$addItemToList = false;
					printf( "Two versions of file found: %s and %s\n", 
						$tmpitem[ "name"], $localitem[ "name"]);
					
					if( (int)$tmpitem[ "svnrev"] > (int)$lpsvnrev)
					{
						printf( "Skipped %s\n", $localitem[ "name"]);
						break;
					}
					else
					{
						printf( "Replaced %s with %s\n", $tmpitem[ "name"], $localitem[ "name"]);
						$tmpitem[ "svnrev"] = $lpsvnrev;
						$tmpitem[ "path"] = $localitem[ "path"];
						$tmpitem[ "name"] = $localitem[ "name"];
						break;
					}
				}
			}
			
			if( $addItemToList == true)
			{
				$lprodlist[ $lprodcount][ "type"] = $lptype;
				$lprodlist[ $lprodcount][ "project"] = $lpprojectname;
				$lprodlist[ $lprodcount][ "ostype"] = $lpostype;
				$lprodlist[ $lprodcount][ "hosttype"] = $lphosttype;
				$lprodlist[ $lprodcount][ "packagearch"] = $lparch;
				$lprodlist[ $lprodcount][ "wordsize"] = $lpwordsize;
				$lprodlist[ $lprodcount][ "svnrev"] = $lpsvnrev;
				$lprodlist[ $lprodcount][ "timestamp"] = $localitem[ "mtime"];
				$lprodlist[ $lprodcount][ "path"] = $localitem[ "path"];
				$lprodlist[ $lprodcount][ "name"] = $localitem[ "name"];
				$lprodcount++;
			}
		}
	}
	
	// Look for items to be updated
	
	for( $i = 0; $i < $lprodcount; $i++)
	{
		$localprod = $lprodlist[ $i];
		$candidateList = array();
		$candidateCount = 0;
		
		for( $j = 0; $j < $rprodcount; $j++)
		{
			$remoteprod = $rprodlist[ $j];
			
			if( $remoteprod[ "type"] == $localprod[ "type"] &&
				 $remoteprod[ "project"] == $localprod[ "project"] &&
				 $remoteprod[ "ostype"] == $localprod[ "ostype"] &&
				 $remoteprod[ "hosttype"] == $localprod[ "hosttype"] &&
				 $remoteprod[ "packagearch"] == $localprod[ "packagearch"] &&
				 $remoteprod[ "wordsize"] == $localprod[ "wordsize"])
			{
				$candidateList[ $candidateCount++] = $remoteprod;
			}
		}
		
		if( $candidateCount > 0)
		{
			printf( "Found %u remote candidate(s) that %s might update.\n", 
				$candidateCount, $localprod[ "name"]);
				
			$uploadDest  = dirname( $candidateList[ 0][ "path"]);
			$uploadDest .= "/";
			$uploadDest .= $localprod[ "name"];
				
			if( $candidateCount == 1 && 
				 (int)$candidateList[ 0][ "svnrev"] < $localprod[ "svnrev"])
			{
				printf( "Local package (%d) newer than remote package (%d).\n",
					$localprod[ "svnrev"], (int)$candidateList[ 0][ "svnrev"]);
				printf( "Uploading local package to '%s' and keeping remote package ... ", $uploadDest);
				mkdir( dirname( $uploadDest), 0777, TRUE);
				copy( $localprod[ "path"], $uploadDest);
				printf( "done.\n");
				$updateCount++;
			}
			else if( $candidateCount > 1)
			{
				// Determine which package on the remote system has the highest
				// svn revision
				
				$svnHighRevItem = null;
				
				foreach( $candidateList as $tmpItem)
				{
					if( $svnHighRevItem == null || 
						 (int)$svnHighRevItem[ "svnrev"] < (int)$tmpItem[ "svnrev"])
					{
						$svnHighRevItem = $tmpItem;
					}
				}
				
				if( $svnHighRevItem[ "svnrev"] < $localprod[ "svnrev"])
				{
					// Remove all candidates except the one with the highest rev
					
					foreach( $candidateList as $tmpItem)
					{
						if( $tmpItem != $svnHighRevItem)
						{
							printf( "Deleting %s ... ", $tmpItem[ "path"]);
							unlink( $tmpItem[ "path"]);
							printf( "done.\n");
							$updateCount++;
						}
					}
					
					// Upload the local item
					
					printf( "Adding %s as %s ... ", $localprod[ "name"], $uploadDest);
					mkdir( dirname( $uploadDest), 0777, TRUE);
					copy( $localprod[ "path"], $uploadDest);
					printf( "done.\n");
					$updateCount++;
				}
				else
				{
					printf( "No update needed.\n");
				}
			}
			else
			{
				printf( "No update needed.\n");
			}
			
			printf( "\n\n");
		}
	}
	
	// Look for items that we have locally but not on the remote system
	
	for( $i = 0; $i < $lprodcount; $i++)
	{
		$localprod = $lprodlist[ $i];
		$foundOnRemoteSystem = false;
		
		for( $j = 0; $j < $rprodcount; $j++)
		{
			$remoteprod = $rprodlist[ $j];
			
			if( $remoteprod[ "type"] == $localprod[ "type"] &&
				 $remoteprod[ "project"] == $localprod[ "project"] &&
				 $remoteprod[ "ostype"] == $localprod[ "ostype"] &&
				 $remoteprod[ "hosttype"] == $localprod[ "hosttype"] &&
				 $remoteprod[ "packagearch"] == $localprod[ "packagearch"] &&
				 $remoteprod[ "wordsize"] == $localprod[ "wordsize"])
			{
				$foundOnRemoteSystem = true;
			}
		}
		
		if( $foundOnRemoteSystem == false)
		{
			if( $localprod[ "type"] == "srcpackage" ||
				 $localprod[ "type"] == "srcrpm")
			{
				$uploadDest  = $remotedirpath;
				$uploadDest .= "/" . $localprod[ "project"];
				$uploadDest .= "/downloads/source";
				$uploadDest .= "/" . $localprod[ "name"];
			}
			else if( $localprod[ "type"] == "rpm" ||
				$localprod[ "type"] == "develrpm")
			{
				if( !translateRPMArch( $localprod[ "packagearch"], 
						$hosttype, $wordsize))
				{
					printf( "Unknown RPM architecture: '%s'\n", $localprod[ "packagearch"]); 
					throw new Exception( "Unable to translate RPM architecture");
				}
				
				$uploadDest  = $remotedirpath;
				$uploadDest .= "/" . $localprod[ "project"];
				$uploadDest .= "/downloads/binaries";
				$uploadDest .= "/" . $localprod[ "ostype"];
				$uploadDest .= "-" . $hosttype;
				$uploadDest .= "-" . $wordsize;
				$uploadDest .= "/" . $localprod[ "name"];
			}
			else if( $localprod[ "type"] == "deb")
			{
				if( !translateDEBArch( $localprod[ "packagearch"], 
						$hosttype, $wordsize))
				{
					printf( "Unknown Debian architecture: '%s'\n", $localprod[ "packagearch"]); 
					throw new Exception( "Unable to translate Debian architecture");
				}
				
				$uploadDest  = $remotedirpath;
				$uploadDest .= "/" . $localprod[ "project"];
				$uploadDest .= "/downloads/binaries";
				$uploadDest .= "/" . $localprod[ "ostype"];
				$uploadDest .= "-" . $hosttype;
				$uploadDest .= "-" . $wordsize;
				$uploadDest .= "/" . $localprod[ "name"];
			}
			else if( $localprod[ "type"] == "binpackage")
			{
				$uploadDest  = $remotedirpath;
				$uploadDest .= "/" . $localprod[ "project"];
				$uploadDest .= "/downloads/binaries";
				$uploadDest .= "/" . $localprod[ "ostype"];
				$uploadDest .= "-" . $localprod[ "hosttype"];
				$uploadDest .= "-" . $localprod[ "wordsize"];
				$uploadDest .= "/" . $localprod[ "name"];
			}
			else
			{
				throw new Exception( "Unknown product type");
			}
			
			printf( "Adding %s as %s ... ", $localprod[ "name"], $uploadDest);
			mkdir( dirname( $uploadDest), 0777, TRUE);
			copy( $localprod[ "path"], $uploadDest);
			printf( "done.\n");
			$updateCount++;
		}
	}
	
	printf( "Updated %u package(s).\n", $updateCount);
	
	// Tell MediaWiki to purge its server-side cache so the changes will
	// appear when users access the downloads page
	
	fclose( fopen("http://www.bandit-project.org/index.php/FLAIM_Download?action=purge", "r"));
	fclose( fopen("http://www.bandit-project.org/index.php/FLAIM_Development_Download?action=purge", "r"));
	
	unset( $sftp);
	unset( $connection);
}

/*******************************************************************************
Desc:
*******************************************************************************/
{
	global $projectlist;
	global $productlist;
	
	while( 1)
	{
		try
		{
			foreach( $projectlist as &$project)
			{
				$svnbaseurl = $project[ "svnbaseurl"];
				$svnsubdir = $project[ "svnsubdir"];
				$projname = $project[ "name"];
				
				printf( "Project name = $projname\n");
				printf( "Checking for repository changes ... ");
				
				$svnrev = 0;
				foreach( $project[ "svnrevdirs"] as $revdir)
				{
					printf( "(%s)", $revdir);
					
					$tmpsvnrev = svnCalcHighRev( $svnbaseurl, $revdir);
					if( (int)$tmpsvnrev > (int)$svnrev)
					{
						$svnrev = $tmpsvnrev;
					}
				}
				
				$project[ "svnrev"] = $svnrev;
				$prevrev = $project[ "prevrev"];
				
				printf( "done.\n\n");
				
				printf( "Previous SVN revision = $prevrev\n");
				printf( "Current SVN revision = $svnrev\n\n");
				
				if( $svnrev == $prevrev)
				{
					printf( "Project repository has not changed.  No build needed.\n");
					continue;
				}
				
				$updateSite = true;
				
				foreach( $productlist as $product)
				{
					if( $product[ "project"] == $project[ "name"])
					{
						if( !buildProduct( $product))
						{
							$updateSite = false;
						}
					}
				}
				
				if( $updateSite == true)
				{
					printf( "Build of %s completed.\n", $project[ "name"]);
					updateFTPServer( $project[ "productdir"]);
				}
				
				$project[ "prevrev"] = $svnrev;
			}
		}
		catch (Exception $e)
		{
			printf( "Caught exception: %s\n", $e->getMessage());
		}
		
		for( $loop = 300; $loop >= 0; $loop--)
		{
			printf( "Sleeping ... %03d\r", $loop);
			sleep( 1);
		}
		
		printf( "\n");
	}
}

?>
