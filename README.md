# Taiwanese Bankruptcy Prediction

**Autor:** Angel Francisco García Guzmán — A01704203

## Contexto

Este proyecto construye un modelo de clasificación binaria para predecir si una empresa entrará en bancarrota a partir de sus indicadores financieros.

**Dataset:** *Taiwanese Bankruptcy Prediction* — recopilado del *Taiwan Economic Journal* durante el periodo 1999–2009. La definición de bancarrota se basa en las regulaciones de la *Taiwan Stock Exchange*.

| Característica | Valor |
|---|---|
| Tipo de problema | Clasificación binaria |
| Área | Negocios / Finanzas |
| Instancias | 6,819 empresas |
| Features | 95 indicadores financieros |
| Tipo de feature | Numérico, normalizado en el rango [0, 1] |
| Variable objetivo | `Bankrupt?` — 0 = no bancarrota, 1 = bancarrota |

## EDA

Los resultados del análisis exploratorio de datos pueden encontrarse en el archivo `notebooks/eda.ipynb`.

## Datos preprocesados

El notebook utilizado para el preprocesamiento se encuentra en `notebooks/preprocessing.ipynb`.

Los resultados del preprocesamiento pueden encontrarse en la carpeta `split-dataset/preprocessed`.
