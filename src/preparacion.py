# -*- coding: utf-8 -*-
"""
Carga y preparación del conjunto de datos.

Decisiones que se toman aquí y que están justificadas en el informe:

1. Se descartan las compras ágiles (Tipo = AG) porque no tienen etiqueta, y
   los convenios marco (Tipo = CM) porque son un mecanismo con catálogo
   cerrado que nunca puede ser trato directo. Además, los convenios marco son
   las únicas líneas sin unidad de medida, rubro ni categoría, de modo que el
   indicador de dato faltante actuaba como un sustituto exacto de la columna
   Tipo, que es una de las que se eliminan por fuga.
2. Se eliminan las columnas que filtran la etiqueta. Tipo, DescripcionTipoOC
   y EsCompraAgil determinan EsTratoDirecto al 100%; un modelo que las use
   alcanza un desempeño perfecto sin aprender nada del problema.
3. La partición se hace por orden de compra y no por línea. El archivo tiene
   102.642 líneas repartidas en 43.548 órdenes, por lo que partir al azar por
   línea dejaría líneas de la misma orden en entrenamiento y en prueba.
"""
import numpy as np
import pandas as pd

from config import (ARCHIVO_DATOS, CATEGORICAS_ALTA, CATEGORICAS_BAJA,
                    EXCLUIR_TIPOS,
                    COLUMNA_FECHA, COLUMNA_GRUPO, NUMERICAS, OBJETIVO,
                    SEMILLA, SEPARADOR, TAM_PRUEBA, TAM_VALIDACION,
                    VALOR_POSITIVO)


def cargar_crudo():
    """Lee el CSV tal como viene, sin tocar nada."""
    return pd.read_csv(ARCHIVO_DATOS, sep=SEPARADOR, low_memory=False)


def filtrar_universo(df):
    """
    Deja solo las filas comparables y crea la columna objetivo.

    Se excluyen dos mecanismos. Las compras ágiles (AG) no tienen etiqueta.
    Los convenios marco (CM) sí la tienen, siempre con valor negativo, pero
    son el único bloque sin unidad de medida, rubro ni categoría: dejarlos
    dentro reintroduce por la puerta de atrás la misma fuga que se elimina al
    sacar la columna Tipo, porque el indicador de dato faltante los identifica
    de forma exacta.
    """
    d = df[~df['Tipo'].isin(EXCLUIR_TIPOS)].copy()
    d['y'] = (d[OBJETIVO] == VALOR_POSITIVO).astype(int)
    return d


def construir_variables(d):
    """
    Construye la matriz de variables.

    Las categóricas de baja cardinalidad quedan como texto para que el
    codificador one-hot las tome después. Las de alta cardinalidad se
    reemplazan por su frecuencia relativa, que es una codificación estable
    y que no depende de la etiqueta, así que no introduce fuga.
    """
    X = pd.DataFrame(index=d.index)

    # Numéricas originales
    for col in NUMERICAS:
        X[col] = pd.to_numeric(d[col], errors='coerce')

    # Transformación logarítmica de los montos: la distribución es muy
    # asimétrica (el máximo es 107 mil millones y la mediana 2 millones).
    for col in ['MontoTotalOC_PesosChilenos', 'TotalNetoOC', 'precioNeto',
                'totalLineaNeto', 'cantidad', 'Impuestos']:
        X['log_' + col] = np.log1p(X[col].clip(lower=0))

    # Razones que describen la estructura de la compra
    X['razon_linea_total'] = X['totalLineaNeto'] / X['TotalNetoOC'].replace(0, np.nan)
    X['razon_impuesto'] = X['Impuestos'] / X['MontoTotalOC_PesosChilenos'].replace(0, np.nan)

    # Fecha: del mes de agosto solo se puede extraer el calendario interno
    fecha = pd.to_datetime(d[COLUMNA_FECHA], errors='coerce')
    X['dia_del_mes'] = fecha.dt.day
    X['dia_semana'] = fecha.dt.dayofweek
    X['es_fin_de_semana'] = (fecha.dt.dayofweek >= 5).astype(int)

    # Tamaño de la orden en número de líneas
    lineas = d.groupby(COLUMNA_GRUPO)[COLUMNA_GRUPO].transform('size')
    X['lineas_en_la_orden'] = lineas

    # Categóricas de baja cardinalidad
    for col in CATEGORICAS_BAJA:
        X[col] = d[col].astype('string').fillna('SIN_DATO')

    # Alta cardinalidad codificada por frecuencia
    for col in CATEGORICAS_ALTA:
        frecuencia = d[col].value_counts(normalize=True)
        X['frec_' + col] = d[col].map(frecuencia).fillna(0.0)

    # Indicadores de dato faltante: el informe de la fase 1 mostró que los
    # nulos no están repartidos al azar, así que su ausencia es informativa.
    for col in ['RegionProveedor', 'RubroN1', 'Categoria', 'UnidadMedida']:
        X['falta_' + col] = d[col].isna().astype(int)

    return X


def particionar(X, y, grupos):
    """
    Divide en entrenamiento, validación y prueba respetando la orden de
    compra, de modo que ninguna orden quede repartida entre conjuntos.
    """
    from sklearn.model_selection import GroupShuffleSplit

    separador = GroupShuffleSplit(n_splits=1, test_size=TAM_PRUEBA,
                                  random_state=SEMILLA)
    idx_resto, idx_prueba = next(separador.split(X, y, groups=grupos))

    X_resto, y_resto = X.iloc[idx_resto], y.iloc[idx_resto]
    grupos_resto = grupos.iloc[idx_resto]

    separador2 = GroupShuffleSplit(n_splits=1, test_size=TAM_VALIDACION,
                                   random_state=SEMILLA)
    idx_entrena, idx_valida = next(
        separador2.split(X_resto, y_resto, groups=grupos_resto))

    return {
        'X_entrena': X_resto.iloc[idx_entrena],
        'y_entrena': y_resto.iloc[idx_entrena],
        'X_valida': X_resto.iloc[idx_valida],
        'y_valida': y_resto.iloc[idx_valida],
        'X_prueba': X.iloc[idx_prueba],
        'y_prueba': y.iloc[idx_prueba],
    }


def preparar():
    """Punto de entrada: devuelve los conjuntos ya separados."""
    crudo = cargar_crudo()
    d = filtrar_universo(crudo)
    X = construir_variables(d)
    y = d['y']
    grupos = d[COLUMNA_GRUPO]
    partes = particionar(X, y, grupos)
    partes['crudo'] = crudo
    partes['universo'] = d
    partes['X'] = X
    partes['y'] = y
    return partes


def columnas_por_tipo(X):
    """Separa las columnas numéricas de las categóricas de texto."""
    categoricas = [c for c in X.columns if str(X[c].dtype) in ('string', 'object')]
    numericas = [c for c in X.columns if c not in categoricas]
    return numericas, categoricas
