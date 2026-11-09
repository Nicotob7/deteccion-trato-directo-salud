@echo off
cd /d "%~dp0.."
call venv\Scripts\activate.bat
echo.
echo  Aplicacion en http://127.0.0.1:5000/
echo.
python app\app.py
pause
