# -*- coding: utf-8 -*-
"""
Tres arquitecturas de aprendizaje profundo para el mismo problema.

Las tres reciben la misma matriz ya preprocesada, de modo que la
comparación con los modelos clásicos sea justa. Lo que cambia entre ellas
es la profundidad, la regularización y la forma de combinar las señales.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from config import MODELOS, SEMILLA, TABLAS

EPOCAS = 30
LOTE = 512


def _keras():
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEMILLA)
    tf.config.experimental.enable_op_determinism()
    return tf


def red_mlp_simple(n_entradas, tf):
    """Arquitectura 1: perceptrón multicapa pequeño, la línea base."""
    k = tf.keras
    modelo = k.Sequential([
        k.layers.Input(shape=(n_entradas,)),
        k.layers.Dense(64, activation='relu'),
        k.layers.Dense(32, activation='relu'),
        k.layers.Dense(1, activation='sigmoid'),
    ], name='mlp_simple')
    return modelo


def red_mlp_profunda(n_entradas, tf):
    """
    Arquitectura 2: red más profunda con normalización por lotes y dropout.

    Se agrega regularización porque la clase positiva es minoritaria y una
    red profunda sin control memoriza el conjunto de entrenamiento.
    """
    k = tf.keras
    modelo = k.Sequential([
        k.layers.Input(shape=(n_entradas,)),
        k.layers.Dense(256, activation='relu'),
        k.layers.BatchNormalization(),
        k.layers.Dropout(0.3),
        k.layers.Dense(128, activation='relu'),
        k.layers.BatchNormalization(),
        k.layers.Dropout(0.3),
        k.layers.Dense(64, activation='relu'),
        k.layers.Dropout(0.2),
        k.layers.Dense(1, activation='sigmoid'),
    ], name='mlp_profunda')
    return modelo


def red_residual(n_entradas, tf):
    """
    Arquitectura 3: bloques con conexión residual.

    La conexión de salto permite que el gradiente llegue a las primeras
    capas sin degradarse, que es el motivo por el que este tipo de bloque
    se usa en redes profundas (He et al., 2016).
    """
    k = tf.keras
    entrada = k.layers.Input(shape=(n_entradas,))
    x = k.layers.Dense(128, activation='relu')(entrada)
    for _ in range(2):
        salto = x
        x = k.layers.Dense(128, activation='relu')(x)
        x = k.layers.BatchNormalization()(x)
        x = k.layers.Dropout(0.2)(x)
        x = k.layers.Dense(128)(x)
        x = k.layers.Add()([x, salto])
        x = k.layers.Activation('relu')(x)
    x = k.layers.Dense(32, activation='relu')(x)
    salida = k.layers.Dense(1, activation='sigmoid')(x)
    return k.Model(entrada, salida, name='red_residual')


def entrenar_redes(Xt_ent, y_ent, Xt_val, y_val):
    """Entrena las tres arquitecturas y devuelve la tabla de resultados."""
    tf = _keras()
    k = tf.keras

    # Ponderación de clases: es la estrategia de balanceo que se usa en las
    # redes, porque remuestrear duplicaría el costo de cada época.
    positivos = y_ent.sum()
    negativos = len(y_ent) - positivos
    pesos = {0: 1.0, 1: float(negativos / max(positivos, 1))}
    print('\nEntrenando redes neuronales (peso de la clase positiva: %.1f)'
          % pesos[1])

    arquitecturas = [
        ('mlp_simple', red_mlp_simple),
        ('mlp_profunda', red_mlp_profunda),
        ('red_residual', red_residual),
    ]

    filas = []
    for nombre, constructor in arquitecturas:
        inicio = time.time()
        modelo = constructor(Xt_ent.shape[1], tf)
        modelo.compile(
            optimizer=k.optimizers.Adam(1e-3),
            loss='binary_crossentropy',
            metrics=[k.metrics.AUC(name='auc_pr', curve='PR'),
                     k.metrics.AUC(name='auc_roc')],
        )
        parada = k.callbacks.EarlyStopping(monitor='val_auc_pr', mode='max',
                                           patience=6,
                                           restore_best_weights=True)
        historia = modelo.fit(
            Xt_ent, y_ent, validation_data=(Xt_val, y_val),
            epochs=EPOCAS, batch_size=LOTE, class_weight=pesos,
            callbacks=[parada], verbose=0,
        )
        prob = modelo.predict(Xt_val, batch_size=2048, verbose=0).ravel()
        pred = (prob >= 0.5).astype(int)
        fila = {
            'modelo': nombre,
            'balanceo': 'ponderacion',
            'filas_entrenamiento': len(y_ent),
            'auc_roc': roc_auc_score(y_val, prob),
            'auc_pr': average_precision_score(y_val, prob),
            'precision': precision_score(y_val, pred, zero_division=0),
            'recall': recall_score(y_val, pred, zero_division=0),
            'f1': f1_score(y_val, pred, zero_division=0),
            'segundos': round(time.time() - inicio, 1),
            'parametros': int(modelo.count_params()),
            'epocas_usadas': len(historia.history['loss']),
        }
        filas.append(fila)
        print('  %-14s AUC-PR=%.4f  AUC-ROC=%.4f  F1=%.4f  params=%s  (%.0fs)'
              % (nombre, fila['auc_pr'], fila['auc_roc'], fila['f1'],
                 f"{fila['parametros']:,}".replace(',', '.'), fila['segundos']))
        modelo.save(MODELOS / ('%s.keras' % nombre))
        pd.DataFrame(historia.history).to_csv(
            TABLAS / ('historia_%s.csv' % nombre), index=False)

    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(TABLAS / 'comparacion_redes.csv', index=False,
                          encoding='utf-8')
    return tabla
