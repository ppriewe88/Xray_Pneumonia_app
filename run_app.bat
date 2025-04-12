REM execute this script from power shell by executing the following line
REM Start-Process -FilePath "C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\96_Xray_integration\run_app.bat"

@echo off
REM path to virtual environment. ADAPT TO YOUR LOCAL ENVIRONMENT!
set VENV_DIR=C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\96_Xray_integration\backend\venv

REM 1.1 activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"
echo [%time%] starting virtual environment ...
if errorlevel 1 (
    echo [ERROR] starting virtual environment: FAILURE
    timeout /t 5 >nul
    exit /b 1
)

REM 1.2. confirmation and short sleeping time
echo [%time%] virtual environment running...
timeout /t 5 >nul


REM 2.1 start mlflow server in background (venv context)
cd C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\96_Xray_integration\backend\src\models
echo [%time%] starting MLFlow server on http://127.0.0.1:8080...
start "" "cmd /c mlflow server --host 127.0.0.1 --port 8080"
if errorlevel 1 (
    echo [ERROR] starting MLFlow-server: FAILURE
    timeout /t 10
    exit /b 1
)

REM 2.2. confirmation and short sleeping time
echo [%time%] MLflow-Server running on http://127.0.0.1:8080!
timeout /t 5 >nul


REM 3.1. start api server in background (venv context)
cd C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\96_Xray_integration\backend\src\api
echo [%time%] starting api server on http://127.0.0.1:8000...
start "" "cmd /c python api_server.py"
if errorlevel 1 (
    echo [ERROR] starting API-server: FAILURE
    timeout /t 10
    exit /b 1
)

REM 3.2. confirmation and short sleeping time
echo [%time%] API-Server running on http://127.0.0.1:8000!
timeout /t 5 >nul


REM 4.1. start react app in background
cd C:\Users\pprie\OneDrive\Dokumente\Python_Projekte\96_Xray_integration\frontend
echo [%time%] starting react app http://127.0.0.1:3000...
start "" "cmd /c npm run start"
if errorlevel 1 (
    echo [ERROR] starting react: FAILURE
    timeout /t 10
    exit /b 1
)

REM 4.2. confirmation and short sleeping time
echo [%time%] react app running on http://127.0.0.1:3000!
timeout /t 10 >nul
exit