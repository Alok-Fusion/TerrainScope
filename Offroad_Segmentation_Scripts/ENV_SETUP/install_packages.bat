@echo off
setlocal

set SCRIPT_DIR=%~dp0
set REQUIREMENTS_FILE=%SCRIPT_DIR%..\requirements.txt

echo Activating the Conda environment 'EDU'...
IF "%CONDA_PREFIX%"=="" (
    IF EXIST "%UserProfile%\miniconda3\condabin\conda.bat" (
        call "%UserProfile%\miniconda3\condabin\conda.bat" activate EDU
    ) ELSE IF EXIST "%UserProfile%\Anaconda3\condabin\conda.bat" (
        call "%UserProfile%\Anaconda3\condabin\conda.bat" activate EDU
    ) ELSE (
        echo Could not find conda.bat. Ensure Conda is installed and initialized.
        exit /b 1
    )
) ELSE (
    call conda activate EDU
)

echo Installing project dependencies from %REQUIREMENTS_FILE% ...
python -m pip install --upgrade pip
python -m pip install -r "%REQUIREMENTS_FILE%"

echo.
echo CPU-safe environment setup complete.
echo If you want GPU acceleration, install the matching CUDA PyTorch wheel for your machine after this step.
endlocal
