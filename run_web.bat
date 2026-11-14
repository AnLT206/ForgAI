@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================================
echo   KHOI DONG FORGAI WEB STUDIO (http://127.0.0.1:8000)
echo ============================================================
python -u web/app.py %*
if errorlevel 1 (
    echo.
    echo [Loi] Khong khoi chay duoc Web Studio. Kiem tra lai moi truong Python.
    pause
)
