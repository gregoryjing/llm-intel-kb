@echo off
cd /d "%~dp0"
echo ================================================
echo    llm-intel-kb - AI/LLM Intelligence KB
echo ================================================
echo.
echo Project: %~dp0
echo.
echo [1/3] Activating virtual environment...
call .venv\Scripts\activate.bat
echo Done.
echo.
echo [2/3] Starting Web UI server...
echo.
echo ================================================
echo    Open browser: http://127.0.0.1:8000
echo ================================================
echo.
echo    Press Ctrl+C to stop
echo.
echo ================================================
echo.
llm-intel-kb serve --port 8000
echo.
echo Server stopped.
pause