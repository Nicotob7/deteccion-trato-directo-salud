# -*- coding: utf-8 -*-
"""
Entrenamiento y comparación de los modelos de aprendizaje automático.

Cubre tres familias de modelos clásicos, tres estrategias de balanceo de
clases y tres arquitecturas de aprendizaje profundo. Se ejecuta con:

    python src/modelado.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import MODELOS, SEMILLA, TABLAS
from preparacion import columnas_por_tipo, preparar

UMBRAL = 0.5


def construir_preprocesador(X):
    """
    Imputa, escala las numéricas y codifica las categóricas con one-hot.

    Se limita max_categories para que la dimensionalidad no se dispare:
    el informe de la fase 1 ya advertía que una variable con miles de
    valores distintos hace inviable el one-hot directo.
    """
    numericas, categoricas = columnas_por_tipo(X)
    tuberia_num = Pipeline([
        ('imputar', SimpleImputer(strategy='median')),
        ('escalar', StandardScaler()),
    ])
    tuberia_cat = Pipeline([
        ('imputar', SimpleImputer(strategy='constant', fill_value='SIN_DATO')),
        ('codificar', OneHotEncoder(handle_unknown='infrequent_if_exist',
                                    min_frequency=50, sparse_output=False)),
    ])
    return ColumnTransformer([
        ('num', tuberia_num, numericas),
        ('cat', tuberia_cat, categoricas),
    ])


def aplicar_balanceo(X, y, estrategia):
    """Devuelve los datos remuestreados según la estrategia pedida."""
    if estrategia == 'sin_balanceo':
        return X, y, None
    if estrategia == 'ponderacion':
        return X, y, 'balanced'
    if estrategia == 'smote':
        from imblearn.over_sampling import SMOTE
        X2, y2 = SMOTE(random_state=SEMILLA, k_neighbors=5).fit_resample(X, y)
        return X2, y2, None
    if estrategia == 'submuestreo':
        from imblearn.under_sampling import RandomUnderSampler
        X2, y2 = RandomUnderSampler(random_state=SEMILLA).fit_resample(X, y)
        return X2, y2, None
    raise ValueError('Estrategia desconocida: %s' % estrategia)


def crear_modelo(nombre, peso):
    if nombre == 'regresion_logistica':
        return LogisticRegression(max_iter=2000, class_weight=peso,
                                  random_state=SEMILLA, n_jobs=-1)
    if nombre == 'bosque_aleatorio':
        return RandomForestClassifier(n_estimators=300, max_depth=18,
                                      min_samples_leaf=5, class_weight=peso,
                                      random_state=SEMILLA, n_jobs=-1)
    if nombre == 'gradient_boosting':
        # HistGradientBoosting no acepta class_weight en versiones antiguas,
        # así que la ponderación se pasa como pesos por muestra.
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              max_depth=8,
                                              random_state=SEMILLA)
    raise ValueError(nombre)


def medir(y_real, probabilidad, umbral=UMBRAL):
    prediccion = (probabilidad >= umbral).astype(int)
    return {
        'auc_roc': roc_auc_score(y_real, probabilidad),
        'auc_pr': average_precision_score(y_real, probabilidad),
        'precision': precision_score(y_real, prediccion, zero_division=0),
        'recall': recall_score(y_real, prediccion, zero_division=0),
        'f1': f1_score(y_real, prediccion, zero_division=0),
    }


def entrenar_clasicos(partes):
    """Entrena las 3 familias clásicas con las 4 estrategias de balanceo."""
    X_ent, y_ent = partes['X_entrena'], partes['y_entrena']
    X_val, y_val = partes['X_valida'], partes['y_valida']

    preprocesador = construir_preprocesador(X_ent)
    print('Ajustando el preprocesador...')
    Xt_ent = preprocesador.fit_transform(X_ent)
    Xt_val = preprocesador.transform(X_val)
    print('  matriz de entrenamiento:', Xt_ent.shape)
    joblib.dump(preprocesador, MODELOS / 'preprocesador.joblib')

    estrategias = ['sin_balanceo', 'ponderacion', 'smote', 'submuestreo']
    nombres = ['regresion_logistica', 'bosque_aleatorio', 'gradient_boosting']

    filas = []
    mejor = {'auc_pr': -1}
    for nombre in nombres:
        for estrategia in estrategias:
            inicio = time.time()
            Xb, yb, peso = aplicar_balanceo(Xt_ent, y_ent.values, estrategia)
            modelo = crear_modelo(nombre, peso)
            if nombre == 'gradient_boosting' and estrategia == 'ponderacion':
                proporcion = (yb == 0).sum() / max((yb == 1).sum(), 1)
                pesos = np.where(yb == 1, proporcion, 1.0)
                modelo.fit(Xb, yb, sample_weight=pesos)
            else:
                modelo.fit(Xb, yb)
            prob = modelo.predict_proba(Xt_val)[:, 1]
            metricas = medir(y_val.values, prob)
            metricas.update({'modelo': nombre, 'balanceo': estrategia,
                             'segundos': round(time.time() - inicio, 1),
                             'filas_entrenamiento': len(yb)})
            filas.append(metricas)
            print('  %-20s %-13s AUC-PR=%.4f  AUC-ROC=%.4f  F1=%.4f  (%.0fs)'
                  % (nombre, estrategia, metricas['auc_pr'],
                     metricas['auc_roc'], metricas['f1'], metricas['segundos']))
            if metricas['auc_pr'] > mejor['auc_pr']:
                mejor = dict(metricas)
                joblib.dump(modelo, MODELOS / 'mejor_clasico.joblib')

    tabla = pd.DataFrame(filas)[['modelo', 'balanceo', 'filas_entrenamiento',
                                 'auc_roc', 'auc_pr', 'precision', 'recall',
                                 'f1', 'segundos']]
    tabla.round(4).to_csv(TABLAS / 'comparacion_modelos_clasicos.csv',
                          index=False, encoding='utf-8')
    print('\n  Mejor modelo clásico:', mejor['modelo'], '+', mejor['balanceo'],
          'AUC-PR=%.4f' % mejor['auc_pr'])
    return preprocesador, Xt_ent, Xt_val, tabla, mejor


def main():
    print('=' * 70)
    print('MODELADO: detección de riesgo en compras por trato directo')
    print('=' * 70)
    partes = preparar()
    for clave in ('entrena', 'valida', 'prueba'):
        X, y = partes['X_' + clave], partes['y_' + clave]
        print('  %-8s %6d filas  %5d positivos (%.2f%%)'
              % (clave, len(X), y.sum(), y.mean() * 100))

    preprocesador, Xt_ent, Xt_val, tabla, mejor = entrenar_clasicos(partes)

    import dl
    tabla_dl = dl.entrenar_redes(Xt_ent, partes['y_entrena'].values,
                                 Xt_val, partes['y_valida'].values)

    resumen = pd.concat([tabla.assign(familia='Machine Learning'),
                         tabla_dl.assign(familia='Deep Learning')],
                        ignore_index=True)
    resumen.round(4).to_csv(TABLAS / 'comparacion_todos.csv', index=False,
                            encoding='utf-8')
    (MODELOS / 'mejor.json').write_text(json.dumps(mejor, indent=1,
                                                   ensure_ascii=False),
                                        encoding='utf-8')
    print('\nListo. Tablas en resultados/tablas/')


if __name__ == '__main__':
    main()
