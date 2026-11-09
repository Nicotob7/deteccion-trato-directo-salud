# -*- coding: utf-8 -*-
"""
TallerRiesgo: aplicación web de apoyo a la fiscalización de compras
públicas por trato directo en el sector Salud.

Backend en Flask y frontend en HTML servido por el mismo proceso. Expone
además una API REST, de modo que el modelo pueda consumirse desde otros
sistemas sin pasar por la interfaz.

Se ejecuta con:  python app/app.py
"""
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / 'src'))

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from config import MODELOS, TABLAS

# Las carpetas se declaran de forma explícita para que la aplicación pueda
# ejecutarse desde cualquier directorio de trabajo.
app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / 'templates'),
    static_folder=str(Path(__file__).resolve().parent / 'static'),
)
REGISTRO = RAIZ / 'resultados' / 'registro_predicciones.csv'

# ---------------------------------------------------------------- carga
PREPROCESADOR = joblib.load(MODELOS / 'preprocesador.joblib')
MODELO = joblib.load(MODELOS / 'mejor_clasico.joblib')
UMBRAL = json.loads((MODELOS / 'umbral.json').read_text(encoding='utf-8'))['umbral']
META = json.loads((MODELOS / 'valores_por_defecto.json').read_text(encoding='utf-8'))
MEJOR = json.loads((MODELOS / 'mejor.json').read_text(encoding='utf-8'))

DEFECTOS = META['defectos']
OPCIONES = META['opciones']
COLUMNAS = META['columnas']

# Campos que se piden en el formulario. El resto se completa con el valor
# habitual del conjunto de datos, para que la interfaz sea usable.
CAMPOS_FORMULARIO = [
    ('MontoTotalOC_PesosChilenos', 'Monto total de la orden (CLP)', 'numero'),
    ('TotalNetoOC', 'Total neto de la orden (CLP)', 'numero'),
    ('precioNeto', 'Precio unitario neto (CLP)', 'numero'),
    ('cantidad', 'Cantidad', 'numero'),
    ('totalLineaNeto', 'Total neto de la linea (CLP)', 'numero'),
    ('PromedioCalificacion', 'Calificacion promedio del proveedor', 'numero'),
    ('lineas_en_la_orden', 'Numero de lineas de la orden', 'numero'),
    ('Estado', 'Estado de la orden', 'lista'),
    ('FormaPago', 'Forma de pago', 'lista'),
    ('RegionUnidadCompra', 'Region de la unidad de compra', 'lista'),
    ('RubroN1', 'Rubro', 'lista'),
]

try:
    IMPORTANCIA = pd.read_csv(TABLAS / 'importancia_shap.csv').head(12)
except Exception:
    IMPORTANCIA = pd.DataFrame(columns=['variable', 'shap_medio_abs'])

try:
    EVALUACION = pd.read_csv(TABLAS / 'evaluacion_prueba.csv')
except Exception:
    EVALUACION = pd.DataFrame()

EXPLICADOR = None
try:
    import shap
    EXPLICADOR = shap.TreeExplainer(MODELO)
except Exception:
    EXPLICADOR = None


def nombres_columnas():
    try:
        return list(PREPROCESADOR.get_feature_names_out())
    except Exception:
        return []


NOMBRES = nombres_columnas()


# ------------------------------------------------------------- utilidades
def armar_fila(entrada):
    """Completa una fila con los valores recibidos y los valores por defecto."""
    fila = dict(DEFECTOS)
    for clave, valor in entrada.items():
        if clave not in fila:
            continue
        if isinstance(DEFECTOS.get(clave), float):
            try:
                fila[clave] = float(str(valor).replace('.', '').replace(',', '.'))
            except (TypeError, ValueError):
                pass
        else:
            fila[clave] = str(valor)
    # Variables derivadas que dependen de las que entrega el usuario
    monto = fila.get('MontoTotalOC_PesosChilenos') or 0
    for base in ['MontoTotalOC_PesosChilenos', 'TotalNetoOC', 'precioNeto',
                 'totalLineaNeto', 'cantidad', 'Impuestos']:
        if 'log_' + base in fila:
            fila['log_' + base] = float(np.log1p(max(fila.get(base, 0) or 0, 0)))
    if 'razon_impuesto' in fila and monto:
        fila['razon_impuesto'] = (fila.get('Impuestos', 0) or 0) / monto
    if 'razon_linea_total' in fila and (fila.get('TotalNetoOC') or 0):
        fila['razon_linea_total'] = (fila.get('totalLineaNeto', 0) or 0) / fila['TotalNetoOC']
    return pd.DataFrame([[fila[c] for c in COLUMNAS]], columns=COLUMNAS)


def clasificar(probabilidad):
    if probabilidad >= max(UMBRAL, 0.5):
        return 'alto', 'Revisar con prioridad'
    if probabilidad >= UMBRAL * 0.6:
        return 'medio', 'Revisar si hay capacidad'
    return 'bajo', 'Sin senales de alerta'


def explicar(matriz):
    """Devuelve los factores que más empujan la predicción de esta compra."""
    if EXPLICADOR is None or not NOMBRES:
        return []
    try:
        valores = EXPLICADOR.shap_values(matriz)
        if isinstance(valores, list):
            valores = valores[1]
        valores = np.array(valores)
        if valores.ndim == 3:
            valores = valores[:, :, 1]
        pares = sorted(zip(NOMBRES, valores[0]), key=lambda t: -abs(t[1]))[:6]
        return [{'variable': n.replace('num__', '').replace('cat__', ''),
                 'aporte': round(float(v), 4),
                 'sentido': 'sube' if v > 0 else 'baja'} for n, v in pares]
    except Exception:
        return []


def registrar(probabilidad, nivel, origen):
    """Deja traza de cada predicción para el panel de monitoreo."""
    nuevo = not REGISTRO.exists()
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRO.open('a', newline='', encoding='utf-8') as fichero:
        escritor = csv.writer(fichero)
        if nuevo:
            escritor.writerow(['momento', 'probabilidad', 'nivel', 'origen'])
        escritor.writerow([datetime.now().isoformat(timespec='seconds'),
                           round(float(probabilidad), 6), nivel, origen])


# ----------------------------------------------------------------- rutas
@app.route('/', methods=['GET', 'POST'])
def inicio():
    resultado = None
    enviado = {}
    if request.method == 'POST':
        enviado = request.form.to_dict()
        matriz = PREPROCESADOR.transform(armar_fila(enviado))
        probabilidad = float(MODELO.predict_proba(matriz)[0, 1])
        nivel, accion = clasificar(probabilidad)
        registrar(probabilidad, nivel, 'formulario')
        resultado = {
            'probabilidad': round(probabilidad * 100, 1),
            'nivel': nivel,
            'accion': accion,
            'umbral': round(UMBRAL * 100, 1),
            'factores': explicar(matriz),
        }
    return render_template('inicio.html', campos=CAMPOS_FORMULARIO,
                           opciones=OPCIONES, defectos=DEFECTOS,
                           resultado=resultado, enviado=enviado,
                           modelo=MEJOR)


@app.route('/api/evaluar', methods=['POST'])
def api_evaluar():
    datos = request.get_json(silent=True) or {}
    matriz = PREPROCESADOR.transform(armar_fila(datos))
    probabilidad = float(MODELO.predict_proba(matriz)[0, 1])
    nivel, accion = clasificar(probabilidad)
    registrar(probabilidad, nivel, 'api')
    return jsonify({'probabilidad': probabilidad, 'nivel': nivel,
                    'accion': accion, 'umbral': UMBRAL,
                    'factores': explicar(matriz)})


@app.route('/monitoreo')
def monitoreo():
    historial = pd.DataFrame()
    if REGISTRO.exists():
        historial = pd.read_csv(REGISTRO)
    resumen = {
        'total': len(historial),
        'por_nivel': (historial['nivel'].value_counts().to_dict()
                      if len(historial) else {}),
        'probabilidad_media': (round(historial['probabilidad'].mean(), 4)
                               if len(historial) else None),
        'ultima': (historial['momento'].iloc[-1] if len(historial) else None),
    }
    return render_template('monitoreo.html', resumen=resumen,
                           evaluacion=EVALUACION.to_dict('records'),
                           importancia=IMPORTANCIA.to_dict('records'),
                           modelo=MEJOR,
                           historial=historial.tail(20).to_dict('records'))


@app.route('/salud')
def salud():
    return jsonify({'estado': 'ok', 'modelo': MEJOR.get('modelo'),
                    'umbral': UMBRAL})


if __name__ == '__main__':
    print('TallerRiesgo en http://127.0.0.1:5000/')
    app.run(host='127.0.0.1', port=5000, debug=False)
