# -*- coding: utf-8 -*-
"""
Evaluación final sobre el conjunto de prueba y explicabilidad con SHAP.

Se ejecuta después de modelado.py:
    python src/evaluacion.py
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from config import FIGURAS, MODELOS, SEMILLA, TABLAS
from preparacion import columnas_por_tipo, preparar

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'figure.autolayout': True})
AZUL, ROJO, VERDE = '#1f4e79', '#c0392b', '#1e8449'


def nombres_de_columnas(preprocesador, X):
    try:
        return list(preprocesador.get_feature_names_out())
    except Exception:
        return ['v%d' % i for i in range(preprocesador.transform(X[:1]).shape[1])]


def main():
    print('Preparando datos...')
    partes = preparar()
    preprocesador = joblib.load(MODELOS / 'preprocesador.joblib')
    modelo = joblib.load(MODELOS / 'mejor_clasico.joblib')
    mejor = json.loads((MODELOS / 'mejor.json').read_text(encoding='utf-8'))
    print('  modelo evaluado:', mejor['modelo'], '+', mejor['balanceo'])

    X_prueba, y_prueba = partes['X_prueba'], partes['y_prueba'].values
    Xt_prueba = preprocesador.transform(X_prueba)
    prob = modelo.predict_proba(Xt_prueba)[:, 1]

    # ---------- Umbral elegido sobre validación ----------
    Xt_val = preprocesador.transform(partes['X_valida'])
    prob_val = modelo.predict_proba(Xt_val)[:, 1]
    p, r, umbrales = precision_recall_curve(partes['y_valida'].values, prob_val)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    mejor_umbral = float(umbrales[int(np.argmax(f1[:-1]))])
    print('  umbral que maximiza F1 en validación: %.3f' % mejor_umbral)

    filas = []
    for etiqueta, umbral in [('umbral 0,5', 0.5), ('umbral ajustado', mejor_umbral)]:
        pred = (prob >= umbral).astype(int)
        matriz = confusion_matrix(y_prueba, pred)
        vn, fp, fn, vp = matriz.ravel()
        filas.append({
            'configuracion': etiqueta,
            'umbral': round(umbral, 3),
            'auc_roc': roc_auc_score(y_prueba, prob),
            'auc_pr': average_precision_score(y_prueba, prob),
            'precision': vp / max(vp + fp, 1),
            'recall': vp / max(vp + fn, 1),
            'f1': 2 * vp / max(2 * vp + fp + fn, 1),
            'verdaderos_positivos': int(vp), 'falsos_positivos': int(fp),
            'falsos_negativos': int(fn), 'verdaderos_negativos': int(vn),
        })
    resultados = pd.DataFrame(filas)
    resultados.round(4).to_csv(TABLAS / 'evaluacion_prueba.csv', index=False,
                               encoding='utf-8')
    print(resultados[['configuracion', 'umbral', 'auc_roc', 'auc_pr',
                      'precision', 'recall', 'f1']].round(3).to_string(index=False))

    # ---------- Figura: curvas ROC y PR ----------
    fig, ejes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    fpr, tpr, _ = roc_curve(y_prueba, prob)
    ejes[0].plot(fpr, tpr, color=AZUL, lw=1.6,
                 label='AUC = %.3f' % roc_auc_score(y_prueba, prob))
    ejes[0].plot([0, 1], [0, 1], '--', color='#999999', lw=1)
    ejes[0].set_xlabel('Tasa de falsos positivos')
    ejes[0].set_ylabel('Tasa de verdaderos positivos')
    ejes[0].set_title('Curva ROC (conjunto de prueba)')
    ejes[0].legend(fontsize=7)

    pp, rr, _ = precision_recall_curve(y_prueba, prob)
    base = y_prueba.mean()
    ejes[1].plot(rr, pp, color=ROJO, lw=1.6,
                 label='AUC-PR = %.3f' % average_precision_score(y_prueba, prob))
    ejes[1].axhline(base, ls='--', color='#999999', lw=1,
                    label='Azar = %.3f' % base)
    ejes[1].set_xlabel('Recall')
    ejes[1].set_ylabel('Precisión')
    ejes[1].set_title('Curva precisión-recall')
    ejes[1].legend(fontsize=7)

    pred = (prob >= mejor_umbral).astype(int)
    matriz = confusion_matrix(y_prueba, pred)
    imagen = ejes[2].imshow(matriz, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ejes[2].text(j, i, f'{matriz[i, j]:,}'.replace(',', '.'),
                         ha='center', va='center',
                         color='white' if matriz[i, j] > matriz.max() / 2 else 'black')
    ejes[2].set_xticks([0, 1]); ejes[2].set_xticklabels(['Otros', 'Trato directo'])
    ejes[2].set_yticks([0, 1]); ejes[2].set_yticklabels(['Otros', 'Trato directo'])
    ejes[2].set_xlabel('Predicción'); ejes[2].set_ylabel('Valor real')
    ejes[2].set_title('Matriz de confusión\n(umbral %.2f)' % mejor_umbral)
    fig.savefig(FIGURAS / 'fig6_curvas_evaluacion.png', bbox_inches='tight')
    plt.close(fig)
    print('  figura -> fig6_curvas_evaluacion.png')

    # ---------- SHAP ----------
    try:
        import shap
        print('Calculando valores SHAP...')
        nombres = nombres_de_columnas(preprocesador, X_prueba)
        muestra = Xt_prueba[np.random.RandomState(SEMILLA).choice(
            len(Xt_prueba), size=min(2000, len(Xt_prueba)), replace=False)]
        try:
            explicador = shap.TreeExplainer(modelo)
            valores = explicador.shap_values(muestra)
            if isinstance(valores, list):
                valores = valores[1]
            if valores.ndim == 3:
                valores = valores[:, :, 1]
        except Exception:
            fondo = shap.sample(Xt_prueba, 100, random_state=SEMILLA)
            explicador = shap.KernelExplainer(
                lambda z: modelo.predict_proba(z)[:, 1], fondo)
            muestra = muestra[:200]
            valores = explicador.shap_values(muestra, nsamples=100)

        importancia = pd.DataFrame({
            'variable': nombres[:valores.shape[1]],
            'shap_medio_abs': np.abs(valores).mean(axis=0),
        }).sort_values('shap_medio_abs', ascending=False)
        importancia.head(40).round(5).to_csv(
            TABLAS / 'importancia_shap.csv', index=False, encoding='utf-8')

        top = importancia.head(15).iloc[::-1]
        fig, eje = plt.subplots(figsize=(7.2, 3.8))
        eje.barh(range(len(top)), top['shap_medio_abs'], color=VERDE)
        eje.set_yticks(range(len(top)))
        eje.set_yticklabels([t.replace('num__', '').replace('cat__', '')[:46]
                             for t in top['variable']], fontsize=6.5)
        eje.set_xlabel('Valor SHAP medio absoluto')
        eje.set_title('Variables que más influyen en la predicción (SHAP)')
        fig.savefig(FIGURAS / 'fig7_shap_importancia.png', bbox_inches='tight')
        plt.close(fig)
        print('  figura -> fig7_shap_importancia.png')
    except Exception as error:
        print('  AVISO: no se pudo calcular SHAP:', error)

    # ---------- Técnica no supervisada ----------
    print('Agrupando con K-Means...')
    numericas, _ = columnas_por_tipo(X_prueba)
    base_km = np.nan_to_num(preprocesador.transform(partes['X_entrena']))
    km = KMeans(n_clusters=5, random_state=SEMILLA, n_init=10).fit(base_km)
    grupos = km.predict(np.nan_to_num(Xt_prueba))
    perfil = pd.DataFrame({'grupo': grupos, 'y': y_prueba}).groupby('grupo').agg(
        lineas=('y', 'size'), tasa_trato_directo=('y', 'mean'))
    perfil['tasa_trato_directo'] = (perfil['tasa_trato_directo'] * 100).round(2)
    perfil.to_csv(TABLAS / 'kmeans_perfiles.csv', encoding='utf-8')
    print(perfil.to_string())

    fig, eje = plt.subplots(figsize=(5.2, 2.8))
    eje.bar(perfil.index.astype(str), perfil['tasa_trato_directo'], color=AZUL)
    eje.axhline(y_prueba.mean() * 100, ls='--', color=ROJO, lw=1,
                label='Tasa global (%.1f%%)' % (y_prueba.mean() * 100))
    eje.set_xlabel('Conglomerado'); eje.set_ylabel('% de trato directo')
    eje.set_title('Tasa de trato directo por conglomerado (K-Means, k = 5)')
    eje.legend(fontsize=7)
    fig.savefig(FIGURAS / 'fig8_kmeans.png', bbox_inches='tight')
    plt.close(fig)
    print('  figura -> fig8_kmeans.png')

    # ---------- Valores por defecto para la aplicación web ----------
    X_todo = partes['X']
    numericas_todo, categoricas_todo = columnas_por_tipo(X_todo)
    defectos = {}
    for col in numericas_todo:
        defectos[col] = float(pd.to_numeric(X_todo[col], errors='coerce').median())
    for col in categoricas_todo:
        moda = X_todo[col].mode()
        defectos[col] = str(moda.iloc[0]) if len(moda) else 'SIN_DATO'
    opciones = {col: sorted(X_todo[col].dropna().astype(str).unique().tolist())[:60]
                for col in categoricas_todo}
    (MODELOS / 'valores_por_defecto.json').write_text(
        json.dumps({'defectos': defectos, 'opciones': opciones,
                    'columnas': list(X_todo.columns)},
                   indent=1, ensure_ascii=False), encoding='utf-8')

    (MODELOS / 'umbral.json').write_text(
        json.dumps({'umbral': mejor_umbral}, indent=1), encoding='utf-8')
    print('\nEvaluación terminada.')


if __name__ == '__main__':
    main()
