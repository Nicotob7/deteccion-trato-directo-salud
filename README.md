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

Tras descartar las compras ágiles, que son las que no tienen etiqueta, el
universo de trabajo queda en **102.642 líneas repartidas en 43.548 órdenes de
compra**, con un **8,08 % de tratos directos**.

### Dos decisiones que condicionan todo el modelado

1. **Se excluyen las columnas que filtran la etiqueta.** `Tipo`,
   `DescripcionTipoOC` y `EsCompraAgil` determinan `EsTratoDirecto` al 100 %.
   Un modelo que las use alcanza un desempeño perfecto sin aprender nada del
   problema.
2. **La partición se hace por orden de compra, no por línea.** Cada orden
   tiene en promedio 2,36 líneas. Repartir al azar por línea dejaría líneas de
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
```

### Aplicación web

```bash
python app/app.py
```

Después abrir <http://127.0.0.1:5000/>. La aplicación tiene tres pantallas:

| Ruta | Qué hace |
|------|----------|
| `/` | Formulario para evaluar una compra. Devuelve la probabilidad, el nivel de riesgo y los factores que explican ese resultado con SHAP. |
| `/monitoreo` | Panel con las métricas del modelo, el uso del servicio y las variables más influyentes. |
| `/salud` | Estado del servicio en JSON. |
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
- `mlp_profunda`: cuatro capas con normalización por lotes y dropout
- `red_residual`: bloques con conexión residual

La métrica principal es **AUC-PR**, porque con un 8 % de positivos la exactitud
no distingue entre un modelo útil y uno que responde siempre "no".

## Resultados

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
