# Detección temprana de riesgo en compras por trato directo del sector Salud

Proyecto del curso **Aprendizaje de Máquinas (ACIF104)** — Universidad Andrés
Bello, Ingeniería Civil Informática. Fase 2.

El Estado de Chile compra a través de Mercado Público. Una parte de esas
compras no pasa por licitación, sino que se adjudica de forma directa a un
proveedor. Ese mecanismo, el **trato directo**, es excepcional por ley, pero
hoy cuenta con un único control automatizado y de tipo formal. Este proyecto
entrena modelos de aprendizaje automático que, a partir de las
características de la propia compra, estiman qué tan parecida es al perfil de
un trato directo, de modo que la fiscalización pueda priorizar qué revisar.

## Qué hay en este repositorio

```
.
├── src/
│   ├── config.py         Rutas, columnas y parámetros del proyecto
│   ├── preparacion.py    Carga, limpieza, variables y partición por orden
│   ├── eda.py            Análisis exploratorio: figuras y tablas
│   ├── modelado.py       3 modelos clásicos x 4 estrategias de balanceo
│   ├── dl.py             3 arquitecturas de aprendizaje profundo
│   └── evaluacion.py     Métricas sobre datos no vistos, SHAP y K-Means
├── app/
│   ├── app.py            Backend Flask + API REST
│   └── templates/        Frontend
├── resultados/
│   ├── figuras/          Gráficos generados
│   ├── tablas/           Métricas y tablas en CSV
│   └── modelos/          Modelos entrenados
├── requirements.txt
└── EJECUTAR.bat          Ejecuta todo el pipeline en Windows
```

## Datos

El conjunto de datos son las órdenes de compra del sector Salud publicadas en
Mercado Público durante agosto de 2026: **137.015 filas y 25 columnas**.

El archivo **no se versiona** en el repositorio por su tamaño (74,6 MB). Para
reproducir el proyecto hay que dejarlo en `data/oc_salud_2026_08.csv`. El
separador de campos es `;`.

Tras descartar las compras ágiles y los convenios marco, el universo de
trabajo queda en **86.829 líneas repartidas en 38.787 órdenes de compra**, con
un **9,55 % de tratos directos**.

### Tres decisiones que condicionan todo el modelado

1. **Se excluyen las columnas que filtran la etiqueta.** `Tipo`,
   `DescripcionTipoOC` y `EsCompraAgil` determinan `EsTratoDirecto` al 100 %.
   Un modelo que las use alcanza un desempeño perfecto sin aprender nada del
   problema.
2. **Se excluyen también los convenios marco.** Una primera versión los dejó
   dentro y obtuvo un AUC-PR de 0,793. Al revisar los valores SHAP apareció que
   la variable más influyente era *que faltara la unidad de medida*, con un peso
   casi triple al de la siguiente. La unidad de medida, el rubro y la categoría
   faltan exactamente en las 15.813 líneas de convenio marco, y ninguna de ellas
   es trato directo: el indicador de dato faltante era un sustituto exacto de la
   columna `Tipo`. Al corregirlo el AUC-PR bajó a 0,743, y esa diferencia es lo
   que la fuga estaba inflando.
3. **La partición se hace por orden de compra, no por línea.** Cada orden
   tiene en promedio 2,24 líneas. Repartir al azar por línea dejaría líneas de
   la misma orden en entrenamiento y en prueba, e inflaría artificialmente las
   métricas.

## Instalación y ejecución

Requisitos: **Python 3.10 o superior** y unos 2 GB libres en disco.

### Opción rápida en Windows

1. Copiar `oc_salud_2026_08.csv` dentro de la carpeta `data/`.
2. Doble clic en `EJECUTAR.bat`.

El archivo crea el entorno virtual, instala las dependencias y corre las tres
etapas en orden. El detalle queda en `log_ejecucion.txt`.

### Paso a paso, en cualquier sistema

```bash
python -m venv venv
venv\Scripts\activate          # en Linux o macOS: source venv/bin/activate
pip install -r requirements.txt

python src/eda.py              # análisis exploratorio
python src/modelado.py         # entrena los modelos clásicos y las redes
python src/evaluacion.py       # evalúa, explica con SHAP y agrupa con K-Means
python src/frecuencias.py      # tablas de frecuencia que usa la aplicación
```

### Aplicación web

```bash
python app/app.py
```

Después abrir <http://127.0.0.1:5000/>. La aplicación expone cuatro rutas:

| Ruta | Qué hace |
|------|----------|
| `/` | Formulario para evaluar una compra. Devuelve la probabilidad, el nivel de riesgo y los factores que explican ese resultado con SHAP. |
| `/monitoreo` | Panel con las métricas del modelo, el uso del servicio, el desvío de la probabilidad media respecto de la tasa base y las variables más influyentes. |
| `/salud` | Estado del servicio, modelo cargado y umbral vigente, en JSON. |
| `/api/evaluar` | API REST. Recibe un JSON con los campos de la compra y devuelve la evaluación. |

Ejemplo de llamada a la API:

```bash
curl -X POST http://127.0.0.1:5000/api/evaluar \
     -H "Content-Type: application/json" \
     -d '{"MontoTotalOC_PesosChilenos": 12000000, "precioNeto": 850000, "cantidad": 2}'
```

## Modelos

**Clásicos**, cada uno con cuatro estrategias frente al desbalance de clases
(sin balanceo, ponderación, SMOTE y submuestreo aleatorio):

- Regresión logística
- Bosque aleatorio
- Gradient boosting por histogramas

**Aprendizaje profundo**, con ponderación de clases:

- `mlp_simple`: perceptrón multicapa de dos capas ocultas
- `mlp_profunda`: tres capas ocultas de 256, 128 y 64 unidades, con
  normalización por lotes y dropout
- `red_residual`: bloques con conexión residual

La métrica principal es **AUC-PR**, porque con un 9,55 % de positivos la
exactitud no distingue entre un modelo útil y uno que responde siempre "no".

## Resultados

El mejor modelo es la **potenciación del gradiente sin ajuste de balanceo**.
Sobre el conjunto de prueba, con 17.798 líneas que ningún modelo vio durante el
entrenamiento ni durante la selección:

| Umbral | AUC-ROC | AUC-PR | Precisión | Recall | F1 |
|--------|---------|--------|-----------|--------|-----|
| 0,500 (por omisión) | 0,934 | 0,743 | 0,881 | 0,490 | 0,630 |
| 0,231 (ajustado en validación) | 0,934 | 0,743 | 0,693 | 0,646 | 0,669 |

Dos observaciones del proceso:

- Las tres arquitecturas de aprendizaje profundo quedaron por debajo de los
  modelos de árboles. La mejor, `red_residual`, alcanzó 0,658 de AUC-PR frente a
  0,723 de la potenciación del gradiente.
- **Ninguna estrategia de balanceo mejoró el AUC-PR.** La ponderación, SMOTE y
  el submuestreo mejoran el F1 medido con umbral fijo en 0,5, pero no cambian el
  orden en que el modelo clasifica los casos. El mismo efecto se consigue
  bajando el umbral, sin generar observaciones sintéticas ni descartar datos
  reales.

Las métricas se generan al ejecutar el pipeline y quedan en
`resultados/tablas/`:

- `comparacion_modelos_clasicos.csv` — las 12 combinaciones de modelo y balanceo
- `comparacion_redes.csv` — las tres arquitecturas de aprendizaje profundo
- `comparacion_todos.csv` — todo junto
- `evaluacion_prueba.csv` — desempeño sobre datos no vistos
- `importancia_shap.csv` — variables más influyentes
- `kmeans_perfiles.csv` — conglomerados y su tasa de trato directo

## Licencia y uso

Trabajo académico. Los datos provienen de la plataforma pública de
Mercado Público y se usan solo con fines educativos.
