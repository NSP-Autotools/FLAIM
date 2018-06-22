@ECHO OFF

setlocal

set solution=flaim
set operation=Build
set build=Release
set platform=Win32
set program=%0

:next_arg
shift
if "%0" == ""           goto do_build
if "%0" == "clean"      ((set operation=Clean)  && goto next_arg)
if "%0" == "Clean"      ((set operation=%0)     && goto next_arg)
if "%0" == "build"      ((set operation=Build)  && goto next_arg)
if "%0" == "Build"      ((set operation=%0%)    && goto next_arg)
if "%0" == "debug"      ((set build=Debug)      && goto next_arg)
if "%0" == "Debug"      ((set build=%0%)        && goto next_arg)
if "%0" == "release"    ((set build=Release)    && goto next_arg)
if "%0" == "Release"    ((set build=%0%)        && goto next_arg)
if "%0" == "win32"      ((set platform=Win32)   && goto next_arg)
if "%0" == "Win32"      ((set platform=%0%)     && goto next_arg)
if "%0" == "32"         ((set platform=Win32)   && goto next_arg)
if "%0" == "win64"      ((set platform=x64)     && goto next_arg)
if "%0" == "Win64"      ((set platform=x64)     && goto next_arg)
if "%0" == "64"         ((set platform=x64)     && goto next_arg)
if "%0" == "x64"        ((set platform=%0%)     && goto next_arg)
if "%0" == "flaim"      ((set solution=flaim)   && goto next_arg)
if "%0" == "xflaim"     ((set solution=xflaim)  && goto next_arg)
if "%0" == "sql"        ((set solution=sql)     && goto next_arg)
if "%0" == "all"        ((set solution=all)     && goto next_arg)
goto help

:do_build
pushd ..\ftk\win32\ftk
call build.cmd %operation% %build% %platform%
popd

if "%solution%" == "all" goto build_all

pushd ..\%solution%\win32\%solution%
call build.cmd %operation% %build% %platform%
popd

goto done

:build_all
pushd ..\flaim\win32\flaim
call build.cmd %operation% %build% %platform%
popd

pushd ..\xflaim\win32\xflaim
call build.cmd %operation% %build% %platform%
popd

pushd ..\sql\win32\sql
call build.cmd %operation% %build% %platform%
popd

goto done

:help
echo Usage: %program% [Build^|Clean] [Release^|Debug] [[Win|x]32^|[Win|x]64] [flaim^|xflaim^|sql^|all]
echo Builds the "%build%|%platform%" configuration of only the ftk and flaim projects by default.

:done
endlocal

