@echo off
setlocal
rem ------------------------------------------------------------------
rem  Illust similarity - local launcher (Windows)
rem    start.bat             : run with all metrics
rem    start.bat --no-pixai  : skip PixAI Tagger (much faster on CPU)
rem  First run creates .venv, installs dependencies and downloads
rem  models (~4.5GB into %USERPROFILE%\.cache). Later runs start in ~30s.
rem ------------------------------------------------------------------
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"

if exist "%PY%" goto :deps
echo [setup] Creating virtual environment...
where py >nul 2>nul
if errorlevel 1 goto :venv_python
py -3 -m venv "%VENV%"
goto :venv_check
:venv_python
python -m venv "%VENV%"
:venv_check
if not exist "%PY%" (
    echo [error] Could not create .venv. Install Python 3.10 - 3.12 from https://www.python.org/ and retry.
    pause
    exit /b 1
)

:deps

if exist "%VENV%\.deps_installed" goto :run
echo [setup] Installing dependencies ^(first run only, this takes several minutes^)...
"%PY%" -m pip install --upgrade pip
rem NVIDIA GPU があれば CUDA 版 torch を先に入れる（PyPI の Windows 向け torch は CPU 版のため）
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :deps_cpu
echo [setup] NVIDIA GPU detected - installing CUDA build of torch ^(about 3GB^)...
"%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
if errorlevel 1 echo [warn] CUDA torch install failed, falling back to the CPU build.
:deps_cpu
"%PY%" -m pip install -r requirements.txt gradio
if errorlevel 1 (
    echo [error] pip install failed. Check your network / proxy and retry.
    pause
    exit /b 1
)
echo ok> "%VENV%\.deps_installed"

:run

if "%ENABLE_PIXAI%"=="" set "ENABLE_PIXAI=1"
if /i "%~1"=="--no-pixai" set "ENABLE_PIXAI=0"
if "%OPEN_BROWSER%"=="" set "OPEN_BROWSER=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

echo [run] ENABLE_PIXAI=%ENABLE_PIXAI%  ^(use "start.bat --no-pixai" to skip the heavy PixAI Tagger^)
echo [run] Starting app. The browser opens automatically when ready. Press Ctrl+C to stop.
"%PY%" app.py
pause
