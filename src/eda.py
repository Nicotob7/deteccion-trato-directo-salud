# -*- coding: utf-8 -*-
"""
Análisis exploratorio de datos.

Genera las tablas y las figuras del apartado de EDA del informe. Se ejecuta
con: python src/eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import COLUMNA_GRUPO, FIGURAS, NUMERICAS, TABLAS
from preparacion import (cargar_crudo, construir_variables, filtrar_universo,
                         indices_particion)

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9,
                     'axes.titlesize': 10, 'figure.autolayout': True})
AZUL, ROJO = '#1f4e79', '#c0392b'


def guardar(fig, nombre):
    ruta = FIGURAS / nombre
    fig.savefig(ruta, bbox_inches='tight')
    plt.close(fig)
    print('  figura ->', ruta.name)


def main():
    print('Cargando el conjunto de datos...')
    crudo = cargar_crudo()
    print('  crudo:', crudo.shape)

    # ---------- Tabla 1: completitud ----------
    nulos = (crudo.isna().mean() * 100).round(2).sort_values(ascending=False)
    completitud = pd.DataFrame({
        'variable': nulos.index,
        'nulos_pct': nulos.values,
        'tipo': [str(crudo[c].dtype) for c in nulos.index],
        'valores_distintos': [crudo[c].nunique() for c in nulos.index],
    })
    completitud.to_csv(TABLAS / 'completitud.csv', index=False, encoding='utf-8')
    print('  tabla  -> completitud.csv')

    d = filtrar_universo(crudo)
    print('  universo etiquetado:', d.shape,
          '| positivos:', int(d.y.sum()), f'({d.y.mean()*100:.2f}%)')

    # ---------- Figura 1: balance de clases ----------
    fig, ejes = plt.subplots(1, 2, figsize=(7.5, 2.9))
    conteo = d['y'].value_counts().sort_index()
    ejes[0].bar(['Otros mecanismos', 'Trato directo'], conteo.values,
                color=[AZUL, ROJO])
    for i, v in enumerate(conteo.values):
        ejes[0].text(i, v, f'{v:,}'.replace(',', '.'), ha='center', va='bottom')
    ejes[0].set_title('Balance de clases (n = %s)' % f'{len(d):,}'.replace(',', '.'))
    ejes[0].set_ylabel('Líneas de orden de compra')
    ejes[0].set_ylim(0, conteo.max() * 1.18)

    perdida = crudo['Tipo'].value_counts()
    ejes[1].bar(perdida.index, perdida.values, color=AZUL)
    ejes[1].bar(['AG'], [perdida.get('AG', 0)], color='#999999')
    ejes[1].set_title('Filas por tipo de orden\n(AG = compra ágil, sin etiqueta)')
    ejes[1].set_yscale('log')
    guardar(fig, 'fig1_balance_clases.png')

    # ---------- Figura 2: distribución de montos ----------
    fig, ejes = plt.subplots(1, 2, figsize=(7.5, 2.9))
    montos = np.log10(d['MontoTotalOC_PesosChilenos'].clip(lower=1))
    ejes[0].hist([montos[d.y == 0], montos[d.y == 1]], bins=45, stacked=False,
                 color=[AZUL, ROJO], label=['Otros', 'Trato directo'],
                 density=True, alpha=.8)
    ejes[0].set_xlabel('log10 del monto total de la orden (CLP)')
    ejes[0].set_ylabel('Densidad')
    ejes[0].set_title('Distribución del monto por clase')
    ejes[0].legend(fontsize=7)

    precio = np.log10(d['precioNeto'].clip(lower=1))
    ejes[1].hist([precio[d.y == 0], precio[d.y == 1]], bins=45,
                 color=[AZUL, ROJO], label=['Otros', 'Trato directo'],
                 density=True, alpha=.8)
    ejes[1].set_xlabel('log10 del precio unitario neto (CLP)')
    ejes[1].set_title('Distribución del precio unitario por clase')
    ejes[1].legend(fontsize=7)
    guardar(fig, 'fig2_distribucion_montos.png')

    # ---------- Figura 3: outliers ----------
    fig, eje = plt.subplots(figsize=(7.5, 3.0))
    columnas = ['MontoTotalOC_PesosChilenos', 'TotalNetoOC', 'Impuestos',
                'precioNeto', 'totalLineaNeto', 'cantidad']
    datos = [np.log10(pd.to_numeric(d[c], errors='coerce').clip(lower=1).dropna())
             for c in columnas]
    caja = eje.boxplot(datos, vert=True, patch_artist=True, showfliers=True,
                       flierprops={'markersize': 1, 'alpha': .25})
    for parche in caja['boxes']:
        parche.set_facecolor('#dbe5f1')
    eje.set_xticklabels([c.replace('MontoTotalOC_PesosChilenos', 'MontoTotalOC')
                         for c in columnas], rotation=18, ha='right', fontsize=8.5)
    eje.set_ylabel('log10 del valor')
    eje.set_title('Dispersión y valores atípicos de las variables continuas')
    guardar(fig, 'fig3_outliers.png')

    # ---------- Figura 4: correlaciones ----------
    # Se grafican solo las variables con alguna correlación relevante con otra,
    # porque un mapa con las 33 resulta ilegible en el informe impreso.
    # La codificación por frecuencia se ajusta con las filas de entrenamiento,
    # igual que en el modelado, para que la figura muestre las mismas
    # variables que recibe el modelo y no otras.
    indices = indices_particion(d['y'], d[COLUMNA_GRUPO])
    X = construir_variables(d, indices_ajuste=indices['entrena'])
    numericas = [c for c in X.columns
                 if str(X[c].dtype) not in ('string', 'object')]
    matriz_total = X[numericas].corr(numeric_only=True)
    sin_diagonal = matriz_total.where(
        ~np.eye(len(matriz_total), dtype=bool)).abs()
    relevantes = sin_diagonal.max()[sin_diagonal.max() >= 0.3].index.tolist()
    if len(relevantes) < 6:
        relevantes = numericas[:14]
    matriz = matriz_total.loc[relevantes, relevantes]

    fig, eje = plt.subplots(figsize=(7.4, 6.4))
    imagen = eje.imshow(matriz, cmap='RdBu_r', vmin=-1, vmax=1)
    eje.set_xticks(range(len(relevantes)))
    eje.set_xticklabels(relevantes, rotation=55, ha='right', fontsize=7.5)
    eje.set_yticks(range(len(relevantes)))
    eje.set_yticklabels(relevantes, fontsize=7.5)
    for i in range(len(relevantes)):
        for j in range(len(relevantes)):
            valor = matriz.iloc[i, j]
            eje.text(j, i, ('%.2f' % valor).replace('0.', '.').replace('-0', '-'),
                     ha='center', va='center', fontsize=6,
                     color='white' if abs(valor) > 0.6 else '#333333')
    eje.set_title('Correlaciones entre las variables numéricas con |r| ≥ 0,3')
    fig.colorbar(imagen, fraction=0.046, pad=0.04)
    guardar(fig, 'fig4_correlaciones.png')

    # ---------- Figura 5: tasa de trato directo por organismo y rubro ----------
    fig, ejes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    org = d.groupby('OrganismoPublico')['y'].agg(['size', 'mean'])
    org = org[org['size'] >= 300].sort_values('mean', ascending=False).head(10)
    etiquetas = [t[:38] + '…' if len(t) > 38 else t for t in org.index]
    ejes[0].barh(range(len(org)), org['mean'] * 100, color=ROJO)
    ejes[0].set_yticks(range(len(org)))
    ejes[0].set_yticklabels(etiquetas, fontsize=7.5)
    ejes[0].invert_yaxis()
    ejes[0].set_xlabel('% de líneas por trato directo')
    ejes[0].set_title('Organismos con mayor uso de trato directo\n(mínimo 300 líneas)')

    rub = d.groupby('RubroN1')['y'].agg(['size', 'mean'])
    rub = rub[rub['size'] >= 500].sort_values('mean', ascending=False).head(10)
    etiquetas = [t[:38] + '…' if len(t) > 38 else t for t in rub.index]
    ejes[1].barh(range(len(rub)), rub['mean'] * 100, color=AZUL)
    ejes[1].set_yticks(range(len(rub)))
    ejes[1].set_yticklabels(etiquetas, fontsize=7.5)
    ejes[1].invert_yaxis()
    ejes[1].set_xlabel('% de líneas por trato directo')
    ejes[1].set_title('Rubros con mayor uso de trato directo\n(mínimo 500 líneas)')
    guardar(fig, 'fig5_tasa_por_grupo.png')

    # ---------- Tabla 2: estadísticas descriptivas ----------
    descriptivas = d[NUMERICAS].describe(
        percentiles=[.25, .5, .75, .95, .99]).T
    descriptivas['nulos_pct'] = (d[NUMERICAS].isna().mean() * 100).round(2)
    descriptivas['negativos'] = [(pd.to_numeric(d[c], errors='coerce') < 0).sum()
                                 for c in NUMERICAS]
    descriptivas.round(2).to_csv(TABLAS / 'descriptivas.csv', encoding='utf-8')
    print('  tabla  -> descriptivas.csv')

    # ---------- Tabla 3: comparación por clase ----------
    comparacion = d.groupby('y')[NUMERICAS].median().T
    comparacion.columns = ['mediana_otros', 'mediana_trato_directo']
    comparacion['razon'] = (comparacion['mediana_trato_directo'] /
                            comparacion['mediana_otros']).round(2)
    comparacion.round(2).to_csv(TABLAS / 'comparacion_por_clase.csv',
                                encoding='utf-8')
    print('  tabla  -> comparacion_por_clase.csv')

    # ---------- Resumen en texto ----------
    resumen = {
        'filas_crudo': len(crudo),
        'columnas_crudo': crudo.shape[1],
        'filas_universo': len(d),
        'ordenes_unicas': int(d['Codigo'].nunique()),
        'lineas_por_orden_media': round(len(d) / d['Codigo'].nunique(), 2),
        'positivos': int(d.y.sum()),
        'tasa_positivos_pct': round(d.y.mean() * 100, 2),
        'desbalance': round((1 - d.y.mean()) / d.y.mean(), 1),
        'variables_construidas': X.shape[1],
        'duplicados_exactos': int(d.duplicated().sum()),
        'fecha_min': str(pd.to_datetime(d['FechaEnvio']).min().date()),
        'fecha_max': str(pd.to_datetime(d['FechaEnvio']).max().date()),
    }
    pd.Series(resumen).to_csv(TABLAS / 'resumen_eda.csv', header=False,
                              encoding='utf-8')
    print('\nResumen del EDA:')
    for clave, valor in resumen.items():
        print(f'  {clave:28s} {valor}')


if __name__ == '__main__':
    main()
