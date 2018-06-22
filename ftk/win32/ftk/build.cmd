@ECHO OFF

setlocal

set solution=ftk
set operation=Build
set build=Release
set platform=Win32
set program=%0

:next_arg
shift
if "%0" == ""           goto do_version_test
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
goto help

:do_version_test
devenv /? | find "Visual Studio Version 9.0" >NULL
if errorlevel 1 goto do_build
find "# Visual Studio 2008" %solution%.sln >NULL
if errorlevel 1 devenv %solution%.sln /Upgrade

:do_build
devenv %solution%.sln /%operation% "%build%|%platform%"

goto done

:help
echo Usage: %program% [Build^|Clean] [Release^|Debug] [[Win|x]32^|[Win|x]64]
echo Builds the "%build%|%platform%" configuration by default.

:done
endlocal
