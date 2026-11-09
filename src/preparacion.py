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
3. La partición se hace por orden de compra y no por línea. El universo de
   trabajo tiene 86.829 líneas repartidas en 38.787 órdenes, con 2,24 líneas
   por orden en promedio, por lo que partir al azar por línea dejaría líneas
   de la misma orden en entrenamiento y en prueba.
4. La codificación por frecuencia se ajusta únicamente con las filas de
   entrenamiento. Calcularla sobre el conjunto completo filtraría la
   distribución de validación y de prueba hacia el modelo.
5. Las dos variables de región se normalizan antes de codificarlas. El
   universo de trabajo trae 26 etiquetas distintas para 16 regiones en la
   región de la unidad de compra y 44 en la del proveedor, por espacios
   sobrantes y por nombres alternativos de la misma región.
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


REGIONES = {
    'araucania': 'Araucanía', 'de la araucania': 'Araucanía',
    'bio-bio': 'Biobío', 'del biobio': 'Biobío', 'biobio': 'Biobío',
    'coquimbo': 'Coquimbo', 'de coquimbo': 'Coquimbo',
    'los lagos': 'Los Lagos', 'de los lagos': 'Los Lagos',
    'los rios': 'Los Ríos', 'de los rios': 'Los Ríos',
    'magallanes y antartica': 'Magallanes', 'de magallanes y de la antartica': 'Magallanes',
    'maule': 'Maule', 'del maule': 'Maule',
    'metropolitana': 'Metropolitana', 'metropolitana de santiago': 'Metropolitana',
    'aysen del general carlos ibanez del campo': 'Aysén',
    'de antofagasta': 'Antofagasta', 'antofagasta': 'Antofagasta',
    'de arica y parinacota': 'Arica y Parinacota', 'arica y parinacota': 'Arica y Parinacota',
    'de atacama': 'Atacama', 'atacama': 'Atacama',
    'de tarapaca': 'Tarapacá', 'tarapaca': 'Tarapacá',
    'de valparaiso': 'Valparaíso', 'valparaiso': 'Valparaíso',
    "del libertador general bernardo o'higgins": "O'Higgins",
    'del libertador general bernardo o\u00b4higgins': "O'Higgins",
    'nuble': 'Ñuble', 'del nuble': 'Ñuble', 'de nuble': 'Ñuble',
    'bio bio': 'Biobío', 'del bio bio': 'Biobío',
    'de aysen': 'Aysén', 'aysen': 'Aysén', 'puerto aysen': 'Aysén',
    'metropolitano': 'Metropolitana', 'santiago': 'Metropolitana',
}

# Las dieciséis regiones del país, que es el conjunto de valores válidos.
REGIONES_VALIDAS = {
    'Arica y Parinacota', 'Tarapacá', 'Antofagasta', 'Atacama', 'Coquimbo',
    'Valparaíso', 'Metropolitana', "O'Higgins", 'Maule', 'Ñuble', 'Biobío',
    'Araucanía', 'Los Ríos', 'Los Lagos', 'Aysén', 'Magallanes',
}


def normalizar_region(valor):
    """
    Lleva las etiquetas de región del archivo a los 16 nombres reales.

    El archivo mezcla tres formas de escribir la misma región: con y sin el
    prefijo «Región de», con espacios sobrantes al final y con acentos
    distintos. Sin esta limpieza, el codificador one-hot crea una columna
    para «Valparaíso» y otra para «Región de Valparaíso », y reparte entre
    ambas la señal de una sola región. Lo que no corresponde a ninguna de
    las dieciséis regiones del país se agrupa bajo la etiqueta OTRA.
    """
    import unicodedata
    if not isinstance(valor, str):
        return valor
    texto = ' '.join(valor.split())
    plano = ''.join(c for c in unicodedata.normalize('NFD', texto.lower())
                    if unicodedata.category(c) != 'Mn')
    plano = plano.replace('\u00b4', "'").replace('`', "'")
    for prefijo in ('region ', 'region'):
        if plano.startswith(prefijo):
            plano = plano[len(prefijo):].strip()
            break
    normalizada = REGIONES.get(plano, texto)
    # Veintidós líneas traen en este campo algo que no es una región:
    # nombres de empresa, códigos sueltos o puntuación. Se agrupan aparte
    # en lugar de dejarlas como categorías propias.
    return normalizada if normalizada in REGIONES_VALIDAS else 'OTRA'


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


def construir_variables(d, indices_ajuste=None):
    """
    Construye la matriz de variables.

    Las categóricas de baja cardinalidad quedan como texto para que el
    codificador one-hot las tome después. Las de alta cardinalidad se
    reemplazan por su frecuencia relativa, que es una codificación estable
    y que no depende de la etiqueta, así que no introduce fuga de la clase.

    El parámetro indices_ajuste indica con qué filas se calcula esa
    frecuencia. Debe recibir solo las filas de entrenamiento: si se calcula
    sobre el conjunto completo, la distribución de validación y de prueba se
    filtra hacia el modelo.
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

    # Categóricas de baja cardinalidad. Las dos de región se normalizan
    # antes de codificarlas.
    for col in CATEGORICAS_BAJA:
        serie = d[col]
        if col in ('RegionUnidadCompra', 'RegionProveedor'):
            serie = serie.map(normalizar_region)
        X[col] = serie.astype('string').fillna('SIN_DATO')

    # Alta cardinalidad codificada por frecuencia, ajustada solo con las
    # filas de entrenamiento.
    base = d if indices_ajuste is None else d.loc[indices_ajuste]
    for col in CATEGORICAS_ALTA:
        frecuencia = base[col].value_counts(normalize=True)
        X['frec_' + col] = d[col].map(frecuencia).fillna(0.0)

    # Indicadores de dato faltante: el informe de la fase 1 mostró que los
    # nulos no están repartidos al azar, así que su ausencia es informativa.
    for col in ['RegionProveedor', 'RubroN1', 'Categoria', 'UnidadMedida']:
        X['falta_' + col] = d[col].isna().astype(int)

    return X


def indices_particion(y, grupos):
    """
    Devuelve las etiquetas de fila de cada conjunto, respetando la orden de
    compra de modo que ninguna orden quede repartida entre conjuntos.
    """
    from sklearn.model_selection import GroupShuffleSplit

    separador = GroupShuffleSplit(n_splits=1, test_size=TAM_PRUEBA,
                                  random_state=SEMILLA)
    pos_resto, pos_prueba = next(separador.split(y, y, groups=grupos))

    y_resto = y.iloc[pos_resto]
    grupos_resto = grupos.iloc[pos_resto]

    separador2 = GroupShuffleSplit(n_splits=1, test_size=TAM_VALIDACION,
                                   random_state=SEMILLA)
    pos_entrena, pos_valida = next(
        separador2.split(y_resto, y_resto, groups=grupos_resto))

    return {
        'entrena': y_resto.index[pos_entrena],
        'valida': y_resto.index[pos_valida],
        'prueba': y.index[pos_prueba],
    }


def preparar():
    """
    Punto de entrada: devuelve los conjuntos ya separados.

    El orden importa. Primero se decide la partición, después se construyen
    las variables usando solo las filas de entrenamiento para ajustar la
    codificación por frecuencia.
    """
    crudo = cargar_crudo()
    d = filtrar_universo(crudo)
    y = d['y']
    idx = indices_particion(y, d[COLUMNA_GRUPO])
    X = construir_variables(d, indices_ajuste=idx['entrena'])

    partes = {
        'X_entrena': X.loc[idx['entrena']], 'y_entrena': y.loc[idx['entrena']],
        'X_valida': X.loc[idx['valida']], 'y_valida': y.loc[idx['valida']],
        'X_prueba': X.loc[idx['prueba']], 'y_prueba': y.loc[idx['prueba']],
        'crudo': crudo, 'universo': d, 'X': X, 'y': y, 'indices': idx,
    }
    return partes


def columnas_por_tipo(X):
    """Separa las columnas numéricas de las categóricas de texto."""
    categoricas = [c for c in X.columns if str(X[c].dtype) in ('string', 'object')]
    numericas = [c for c in X.columns if c not in categoricas]
    return numericas, categoricas
