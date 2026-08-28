@echo off
rem ==============================================================================
rem Smart AI 27B — 1-Click Windows Launcher
rem ==============================================================================
cd /d "%~dp0"
echo ======================================================================
echo   Launching Smart AI 27B Autonomous Reasoning Studio
echo ======================================================================

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python app_gui.py %*
    goto end
)

where python3 >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    python3 app_gui.py %*
    goto end
)

echo [!] Error: Python not found in system PATH.
pause

:end
