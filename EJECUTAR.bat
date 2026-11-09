@echo off
setlocal
cd /d "%~dp0"
set LOG=%~dp0log_ejecucion.txt
echo === INICIO %DATE% %TIME% > "%LOG%"

set PY=
py -3 --version >nul 2>&1 && set PY=py -3
if "%PY%"=="" (python --version >nul 2>&1 && set PY=python)
if "%PY%"=="" (
  echo No se encontro Python en el PATH. & pause & exit /b 1
)

if not exist venv (
  echo Creando entorno virtual, esto toma un momento...
  %PY% -m venv venv >> "%LOG%" 2>&1
)
call venv\Scripts\activate.bat

echo Instalando dependencias (la primera vez descarga ~500 MB)...
python -m pip install --disable-pip-version-check -q --upgrade pip >> "%LOG%" 2>&1
python -m pip install --disable-pip-version-check -r requirements.txt >> "%LOG%" 2>&1
echo PIP_FIN=%ERRORLEVEL% >> "%LOG%"

if not exist data\oc_salud_2026_08.csv (
  echo.
  echo FALTA EL DATASET: copie oc_salud_2026_08.csv dentro de la carpeta data\
  echo.
  pause
  exit /b 1
)

echo.
echo [1/3] Analisis exploratorio...
python src\eda.py >> "%LOG%" 2>&1
echo EDA_FIN=%ERRORLEVEL% >> "%LOG%"

echo [2/3] Entrenamiento de modelos...
python src\modelado.py >> "%LOG%" 2>&1
echo MODELADO_FIN=%ERRORLEVEL% >> "%LOG%"

echo [3/3] Evaluacion y explicabilidad...
python src\evaluacion.py >> "%LOG%" 2>&1
echo EVAL_FIN=%ERRORLEVEL% >> "%LOG%"

echo.
echo ==========================================================
echo  Listo. Revise resultados\figuras y resultados\tablas
echo  El detalle quedo en log_ejecucion.txt
echo ==========================================================
echo.
pause
