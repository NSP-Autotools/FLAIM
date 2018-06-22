@echo off

REM -------------------------------------------------------------------------
REM Desc:	Batch file for building GNU printf on Windows platforms	
REM Tabs:	3
REM
REM		Copyright (c) 2006 Novell, Inc. All Rights Reserved.
REM
REM		This program is free software; you can redistribute it and/or
REM		modify it under the terms of version 2 of the GNU General Public
REM		License as published by the Free Software Foundation.
REM
REM		This program is distributed in the hope that it will be useful,
REM		but WITHOUT ANY WARRANTY; without even the implied warranty of
REM		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
REM		GNU General Public License for more details.
REM
REM		You should have received a copy of the GNU General Public License
REM		along with this program; if not, contact Novell, Inc.
REM
REM		To contact Novell about this file by physical or electronic mail,
REM		you may find current contact information at www.novell.com
REM
REM $Id$
REM -------------------------------------------------------------------------

setlocal
if exist build-dir rd /s /q build-dir
mkdir build-dir
cd build-dir
cl /nologo -I.. ../*.cpp /Feprintf.exe
cd ..
endlocal
