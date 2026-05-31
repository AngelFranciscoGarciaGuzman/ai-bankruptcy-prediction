# Predicción de Bancarrota en Empresas Taiwanesas

**Autor:** Angel Francisco García Guzmán — A01704203

## Abstract

La predicción temprana de la bancarrota empresarial es un problema clásico en finanzas cuantitativas, con implicaciones directas para inversionistas, reguladores y entidades crediticias. Este proyecto plantea un clasificador binario para predecir si una empresa entrará en bancarrota a partir de 95 indicadores financieros recopilados del *Taiwan Economic Journal* (1999–2009). El presente entregable cubre las fases de obtención y separación del conjunto de datos, el análisis exploratorio (EDA) y el preprocesamiento.

El dataset original contiene 6,819 empresas con un fuerte desbalance de clases (96.77 % no bancarrota, 3.23 % bancarrota). Tras una división estratificada en train/test (80/20), se realizó un análisis exploratorio que identificó tres hallazgos centrales: (1) la presencia de una feature constante (`Net Income Flag`) sin poder predictivo; (2) escalas heterogéneas entre variables, con outliers extremos que alcanzan hasta diez órdenes de magnitud por encima del rango típico; y (3) multicolinealidad fuerte, con 36 pares de features con correlación absoluta superior a 0.9.

A partir de estos hallazgos se diseñó un pipeline de preprocesamiento que (i) elimina la feature constante, (ii) aplica un escalado robusto a outliers mediante `QuantileTransformer` con distribución de salida normal, y (iii) reduce la dimensionalidad con PCA conservando el 95 % de la varianza, transformando las 94 features restantes en 31 componentes ortogonales. Todos los transformadores se ajustan exclusivamente con el conjunto de entrenamiento para evitar fuga de información (*data leakage*). El modelado y la evaluación final se abordarán en una entrega posterior.

## Introducción

La identificación oportuna de empresas en riesgo de bancarrota constituye un objetivo central de la analítica financiera y del análisis de crédito. Modelos clásicos como el Z-score de Altman (1968) demostraron que combinaciones lineales de ratios financieros (rentabilidad, endeudamiento, liquidez y eficiencia) pueden ofrecer poder predictivo significativo. Con la disponibilidad creciente de datos y técnicas de aprendizaje automático, estos enfoques han evolucionado hacia modelos no lineales capaces de capturar interacciones complejas entre indicadores.

El objetivo del presente proyecto es construir un clasificador binario que distinga entre empresas que entran en bancarrota y empresas que no, utilizando indicadores financieros estandarizados como entrada. Esta primera fase del trabajo se enfoca en preparar los datos para la etapa de modelado, asegurando que el preprocesamiento respete dos principios fundamentales: independencia entre los conjuntos de entrenamiento y prueba, y robustez ante las particularidades estadísticas del dataset (desbalance, outliers, multicolinealidad).

## Descripción del Dataset

El conjunto de datos proviene del *UC Irvine Machine Learning Repository* y fue recopilado por Liang et al. (2020) a partir de registros del *Taiwan Economic Journal* entre 1999 y 2009. La definición de bancarrota se basa en las regulaciones de la *Taiwan Stock Exchange*. Contiene 6,819 observaciones de empresas, cada una descrita por 95 indicadores financieros numéricos y una variable objetivo binaria.

| Característica | Valor |
|---|---|
| Tipo de problema | Clasificación binaria |
| Área | Negocios / Finanzas |
| Instancias | 6,819 empresas |
| Features | 95 indicadores financieros |
| Tipo de feature | Numérico (mayoritariamente `float64`, tres `int64`) |
| Variable objetivo | `Bankrupt?` — 0 = no bancarrota, 1 = bancarrota |
| Valores faltantes | 0 |
| Duplicados | 0 |

Las features cubren las principales familias de indicadores utilizados en análisis financiero, entre ellas: rentabilidad (ROA, márgenes), endeudamiento (debt ratios, leverage), liquidez (current ratio, quick ratio), rotación (turnover rates), crecimiento (growth rates) y estructura de capital.

## Metodología

### Separación del conjunto de datos

La división del dataset en conjuntos de entrenamiento y prueba se realizó **antes** de cualquier transformación, mediante `train_test_split` de scikit-learn con `stratify=y` para preservar la proporción de clases en ambos conjuntos. Se utilizó un 80 % para entrenamiento y un 20 % para prueba.

| Conjunto | Filas | No bancarrota | Bancarrota | Proporción de la clase positiva |
|---|---|---|---|---|
| Train | 5,455 | 5,279 | 176 | 3.23 % |
| Test | 1,364 | 1,320 | 44 | 3.23 % |

**Tabla 1.** Distribución estratificada del dataset.

El script correspondiente se encuentra en `scripts/split_dataset.py`. Se enfatiza que, a partir de este punto, el conjunto de prueba no se utiliza en ninguna decisión de preprocesamiento o modelado: únicamente se evaluará una vez al final del proyecto.

### Análisis Exploratorio (EDA)

El análisis exploratorio se documentó en `notebooks/eda.ipynb` y se estructuró en cinco fases: inspección general, análisis del target, análisis univariado, análisis bivariado y análisis multivariado.

#### Distribución del target

La variable objetivo presenta un desbalance considerable: el 96.77 % de las empresas no quiebra y solo el 3.23 % sí, equivalente a una empresa en bancarrota por cada 30 sin bancarrota. Este desbalance refleja la rareza intrínseca del evento en datos reales y no se considera un artefacto del muestreo. Se descarta el uso de técnicas de generación de datos sintéticos (SMOTE, ADASYN) en esta primera fase, optando por estrategias menos invasivas como ponderación de clases en la etapa de modelado.

#### Análisis univariado

Se evaluó la varianza de cada feature y se identificó **una variable constante**: `Net Income Flag`, cuyo único valor es 1 en todas las observaciones. Esta feature no aporta información discriminativa y se marcó para eliminación. Otras nueve features presentan varianza muy baja (inferior a 0.0002), pero sin llegar a ser constantes; se determinó no eliminarlas manualmente, dejando que la reducción de dimensionalidad las descarte si efectivamente no contribuyen.

El análisis de distribuciones (KDE) y outliers (boxplots) sobre un subconjunto representativo de 16 features reveló dos problemas:

- **Escalas heterogéneas:** mientras la mayoría de features se ubica en rangos próximos a [0, 1], algunas (`Quick Ratio`, `Total Asset Growth Rate`, `Operating Expense Rate`, entre otras) alcanzan valores del orden de 10⁹ a 10¹⁰. Esto contradice la suposición inicial de que el dataset venía completamente normalizado.
- **Outliers extremos:** casi todas las features presentan numerosos puntos fuera de los bigotes en los boxplots, con casos particularmente severos en las variables de mayor escala.

#### Análisis bivariado

Para identificar features con poder predictivo, se calculó la diferencia normalizada de medias entre clases. Las features con mayor capacidad discriminativa pertenecen a dos familias económicamente coherentes:

| Feature | Diferencia normalizada |
|---|---|
| Net Income to Total Assets | 1.79 |
| ROA(A) before interest and % after tax | 1.61 |
| ROA(B) before interest and depreciation after tax | 1.56 |
| ROA(C) before interest and depreciation before interest | 1.49 |
| Debt ratio % | 1.40 |
| Net worth/Assets | 1.40 |
| Persistent EPS in the Last Four Seasons | 1.25 |
| Retained Earnings to Total Assets | 1.21 |

**Tabla 2.** Top 8 features con mayor capacidad discriminativa entre clases.

Los resultados son consistentes con la intuición económica: las empresas que quiebran muestran menor rentabilidad (ROA, Net Income) y mayor endeudamiento (Debt ratio, dependencia de préstamos).

#### Análisis multivariado

La matriz de correlación entre las 95 features evidenció una **multicolinealidad fuerte**, con 36 pares de features con correlación absoluta superior a 0.9. Se identificaron varias familias redundantes:

- Tres variantes de ROA (A, B, C) con correlaciones entre sí de 0.94 a 0.99.
- Tres variantes de *Net Value Per Share* (A, B, C) con correlaciones superiores a 0.999.
- Variantes de *Interest Rate* (pre-tax, after-tax, continuous) con correlaciones de 0.99.
- Pares matemáticamente complementarios, como `Debt ratio %` y `Net worth/Assets` (correlación de −1.0, suman 1).
- Columnas duplicadas con nombres distintos, como `Current Liabilities/Equity` y `Current Liability to Equity` (correlación 1.0).

Esta redundancia justifica la aplicación de técnicas de reducción de dimensionalidad.

### Preprocesamiento

El preprocesamiento se implementó en `notebooks/preprocessing.ipynb` y produce datasets transformados en `split-dataset/preprocessed/`, junto con los transformadores ajustados en formato `joblib`.

#### Eliminación de feature constante

Se eliminó la columna `Net Income Flag` de ambos conjuntos (train y test). El número de features se redujo de 95 a 94.

#### Escalado

La elección del scaler se realizó iterativamente. Inicialmente se probó `RobustScaler`, que utiliza mediana e IQR y, en teoría, es resistente a outliers al estimar sus parámetros. Sin embargo, debido a que algunas features presentan IQRs minúsculos pero outliers de magnitud 10⁹ a 10¹⁰, los valores extremos resultaron amplificados después del escalado (alcanzando magnitudes del orden de 10¹²). Como consecuencia, el PCA subsiguiente colapsaba toda la varianza en un único componente, lo cual indicaba que los outliers seguían dominando la transformación.

Se optó entonces por `QuantileTransformer` con `output_distribution='normal'`, que mapea cada valor a su percentil y posteriormente a una distribución normal estándar. Esta transformación acota cualquier outlier extremo a un rango aproximado de ±5, independientemente de su magnitud original. El transformador se ajustó únicamente con el conjunto de entrenamiento (`fit_transform`) y se aplicó al conjunto de prueba sin re-ajustar (`transform`), garantizando la ausencia de fuga de información.

#### Reducción de dimensionalidad

Sobre los datos escalados se aplicó PCA con un criterio de varianza explicada del 95 % (`n_components=0.95`). El algoritmo seleccionó automáticamente **31 componentes**, suficientes para conservar el 95 % de la varianza total de las 94 features originales. La reducción de dimensionalidad equivale a una compresión del 67 %, justificada por la multicolinealidad documentada en el EDA.

Al igual que el scaler, el PCA se ajustó únicamente con el conjunto de entrenamiento y se aplicó al conjunto de prueba sin re-ajustar.

#### Pipeline completo

El orden estricto de las transformaciones es: split estratificado → eliminación de feature constante → escalado con `QuantileTransformer` → PCA. Cada paso se ajusta solo con datos de entrenamiento, y los transformadores ajustados se serializan para reproducibilidad y futuras predicciones.

| Etapa | Train | Test |
|---|---|---|
| Inicial | (5455, 96) | (1364, 96) |
| Drop `Net Income Flag` | (5455, 95) | (1364, 95) |
| Separación X / y | X: (5455, 94), y: (5455,) | X: (1364, 94), y: (1364,) |
| Después de scaling | (5455, 94) | (1364, 94) |
| Después de PCA | (5455, 31) | (1364, 31) |

**Tabla 3.** Evolución de las dimensiones a lo largo del pipeline de preprocesamiento.

## Evaluación del modelo

Antes de entrar al entrenamiento conviene definir cómo vamos a medir si el modelo es bueno o no. Por el desbalance fuerte del dataset (96.77 % no quiebra, 3.23 % quiebra), no podemos usar accuracy como métrica principal: un modelo que siempre diga "no quiebra" sacaría más del 96 % y sería completamente inútil para el problema real, porque nunca detectaría una quiebra.

Por eso vamos a usar métricas que penalizan correctamente los errores en la clase minoritaria. Precision mide, de las empresas que el modelo predice como quiebra, cuántas realmente quiebran. Recall mide lo opuesto: de las empresas que realmente quiebran, cuántas el modelo detecta. F1-score es la media armónica de precision y recall, una métrica resumen que solo es alta cuando ambas lo son, y es el estándar para casos con desbalance. PR-AUC (área bajo la curva precision-recall) mide el desempeño en todos los umbrales de decisión posibles, y es más informativa que ROC-AUC cuando la clase positiva es rara. La matriz de confusión la usamos para ver el desglose visual de falsos positivos y falsos negativos, no solo los números agregados.

La métrica más importante para este problema es el recall, porque equivocarse en una quiebra (predecir "no quiebra" cuando sí va a quebrar) tiene consecuencias más graves que dar una falsa alarma. Un inversor o un banco prefiere recibir varias advertencias de quiebra y descartar algunas, a perderse la única que importa.

### Baseline de referencia

Como punto de comparación tomamos el paper de Pham et al. (2025), que aplicó SVM, Random Forest y ANN sobre exactamente el mismo dataset. Los resultados de F1-score que reportan, con y sin balanceo de clases, son los siguientes.

| Modelo | Sin SMOTE | Con SMOTE-ENN |
|---|---|---|
| SVM (RBF kernel) | 0.018 | 0.963 |
| Random Forest | 0.194 | 0.981 |
| ANN | 0.216 | 0.985 |

**Tabla 4.** F1-score reportado por Pham et al. (2025), con y sin balanceo de clases.

Lo que esta tabla nos dice es clave. Primero, los modelos sin manejo del desbalance son inútiles para este problema: F1-scores muy bajos y recalls cercanos a cero. Esto confirma que no podemos saltarnos la etapa de manejo del desbalance. Segundo, con un pipeline avanzado (resampling SMOTE-ENN, feature selection con Binary PSO y hyperparameter tuning) se pueden alcanzar F1 superiores a 0.95 con las tres arquitecturas.

Nuestro objetivo realista para este proyecto es alcanzar un F1 mayor a 0.90 con un pipeline más simple. Un resultado entre 0.95 y 0.98 sería competitivo con la literatura sobre este dataset.

## Referencias

Liang, D., Lu, C.-C., Tsai, C.-F., & Shih, G.-A. (2020). *Taiwanese Bankruptcy Prediction* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5004D

Pham, H. V., Chu, T., Le, T. M., Tran, H. M., Tran, H. T. K., Yen, K. N., & Dao, S. V. T. (2025). Comprehensive evaluation of bankruptcy prediction in Taiwanese firms using multiple machine learning models. *International Journal of Technology*, 16(1), 289–309. https://doi.org/10.14716/ijtech.v16i1.7227
