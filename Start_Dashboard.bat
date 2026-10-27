@echo off
setlocal
title Credit Risk Workspace
cd /d "%~dp0"
set "PORTFOLIO_APP=%~dp0scripts\chatbot_app.py"

if not exist "%~dp0.venv\Scripts\python.exe" (
    echo The project Python environment is missing.
    echo Please follow docs\ADMIN_SETUP.md to install it.
    pause
    exit /b 1
)

rem Reopen this project's existing dashboard if it is already running.
powershell.exe -NoProfile -Command "$listeners=Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue; foreach ($listener in $listeners) { $appProcess=Get-CimInstance Win32_Process -Filter ('ProcessId='+$listener.OwningProcess) -ErrorAction SilentlyContinue; if ($appProcess.CommandLine -and $appProcess.CommandLine.Contains($env:PORTFOLIO_APP) -and $appProcess.CommandLine.Contains('streamlit')) { Start-Process 'http://127.0.0.1:8501'; exit 0 } }; exit 1"
if not errorlevel 1 (
    echo Your dashboard is already running. Opening it in your browser.
    exit /b 0
)

rem Choose an available local port without stopping other applications.
set "PORTFOLIO_PORT="
for /f %%P in ('powershell.exe -NoProfile -Command "foreach ($port in 8501..8510) { $probe=[System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback,$port); try { $probe.Start(); $probe.Stop(); Write-Output $port; break } catch { $probe.Stop() } }"') do set "PORTFOLIO_PORT=%%P"
if not defined PORTFOLIO_PORT (
    echo No available dashboard port was found between 8501 and 8510.
    pause
    exit /b 1
)

echo Starting the Credit Risk Workspace...
echo Open http://127.0.0.1:%PORTFOLIO_PORT% if the browser does not open automatically.
echo Keep this window open while using the dashboard. Press Ctrl+C to stop.
echo.
"%~dp0.venv\Scripts\python.exe" -m streamlit run "%PORTFOLIO_APP%" --server.address 127.0.0.1 --server.port %PORTFOLIO_PORT% --server.headless false --browser.gatherUsageStats false
if errorlevel 1 (
    echo.
    echo The dashboard could not start. The error is shown above.
    pause
)
endlocal
