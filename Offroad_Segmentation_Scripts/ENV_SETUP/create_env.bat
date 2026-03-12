@echo off

where conda >nul 2>nul
IF ERRORLEVEL 1 (
    echo Conda is not available. Install Miniconda or Anaconda first.
    exit /b 1
)

call conda env list | findstr /R /C:"^EDU " >nul 2>nul
IF %ERRORLEVEL% EQU 0 (
    echo Conda environment 'EDU' already exists.
    exit /b 0
)

echo Creating the Conda environment 'EDU' with Python 3.10...
call conda create --name EDU python=3.10 -y
