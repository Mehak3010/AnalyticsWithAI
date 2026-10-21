@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Project virtual environment not found.
    exit /b 1
)

".venv\Scripts\python.exe" -m anomaly_agent.cli --config "config\dataco_supply_chain.yaml" >> "logs\daily_run.log" 2>&1

exit /b %errorlevel%
