@echo off
setlocal
set SCRIPT_DIR=%~dp0

echo Running create_env.bat...
call "%SCRIPT_DIR%create_env.bat"
IF ERRORLEVEL 1 exit /b 1

echo Running install_packages.bat...
call "%SCRIPT_DIR%install_packages.bat"
IF ERRORLEVEL 1 exit /b 1

echo All tasks completed.
endlocal
