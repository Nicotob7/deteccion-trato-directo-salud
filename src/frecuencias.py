# -*- coding: utf-8 -*-
"""
Guarda las tablas de frecuencia de las variables de alta cardinalidad.

La aplicación web necesita estas tablas para poder ofrecer un selector con
nombres de organismo y de categoría, en lugar de dejar congelada la variable
frec_OrganismoPublico, que es la primera del listado SHAP. Las tablas se
calculan con las mismas filas de entrenamiento que usa el modelo, con la
misma semilla y la misma partición por orden de compra, de modo que el valor
que entrega la aplicación es el mismo valor que vio el modelo al entrenar.

Se ejecuta con:  python src/frecuencias.py
"""
import json

from config import CATEGORICAS_ALTA, COLUMNA_GRUPO, MODELOS
from preparacion import cargar_crudo, filtrar_universo, indices_particion


def construir():
    crudo = cargar_crudo()
    d = filtrar_universo(crudo)
    idx = indices_particion(d['y'], d[COLUMNA_GRUPO])
    entrena = d.loc[idx['entrena']]

    tablas = {}
    for col in CATEGORICAS_ALTA:
        frecuencia = entrena[col].value_counts(normalize=True)
        tablas[col] = {str(k): float(v) for k, v in frecuencia.items()}
    return tablas


def main():
    tablas = construir()
    destino = MODELOS / 'frecuencias.json'
    destino.write_text(json.dumps(tablas, ensure_ascii=False), encoding='utf-8')
    print('tabla -> frecuencias.json')
    for col, tabla in tablas.items():
        print(col, 'categorias:', len(tabla))


if __name__ == '__main__':
    main()
