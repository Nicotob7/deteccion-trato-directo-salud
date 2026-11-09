# -*- coding: utf-8 -*-
"""Configuración central del proyecto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / 'data'
RESULTADOS = RAIZ / 'resultados'
FIGURAS = RESULTADOS / 'figuras'
TABLAS = RESULTADOS / 'tablas'
MODELOS = RESULTADOS / 'modelos'
for carpeta in (DATOS, FIGURAS, TABLAS, MODELOS):
    carpeta.mkdir(parents=True, exist_ok=True)

ARCHIVO_DATOS = DATOS / 'oc_salud_2026_08.csv'
SEPARADOR = ';'
SEMILLA = 42

# Variable objetivo
OBJETIVO = 'EsTratoDirecto'
VALOR_POSITIVO = 'Si'

# Columnas que se excluyen por fuga de información: determinan la etiqueta
# de forma directa, así que un modelo que las use no aprende nada útil.
FUGA = ['Tipo', 'DescripcionTipoOC', 'EsCompraAgil']

# Mecanismos que salen del universo de trabajo.
#   AG = compra ágil: no tiene etiqueta.
#   CM = convenio marco: catálogo cerrado, nunca es trato directo, y es el
#        único bloque sin unidad de medida, rubro ni categoría, por lo que el
#        indicador de dato faltante lo identificaba de forma exacta.
EXCLUIR_TIPOS = ['AG', 'CM']

# Identificadores y texto libre que no entran al modelo
IDENTIFICADORES = ['Codigo', 'Nombre', 'EspecificacionComprador']

NUMERICAS = [
    'MontoTotalOC_PesosChilenos', 'Impuestos', 'TotalNetoOC',
    'cantidad', 'precioNeto', 'totalLineaNeto', 'PromedioCalificacion',
]

CATEGORICAS_BAJA = [
    'Estado', 'FormaPago', 'RegionUnidadCompra', 'RegionProveedor',
    'RubroN1', 'UnidadMedida',
]

# Alta cardinalidad: se codifican por frecuencia, no con one-hot
CATEGORICAS_ALTA = [
    'OrganismoPublico', 'UnidadCompra', 'NombreProveedor', 'Categoria',
]

COLUMNA_GRUPO = 'Codigo'   # la orden de compra: la partición se hace por aquí
COLUMNA_FECHA = 'FechaEnvio'

TAM_PRUEBA = 0.20
TAM_VALIDACION = 0.20
