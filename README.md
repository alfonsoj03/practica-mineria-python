# Práctica: calidad de datos, minería y despliegue en Python

Repositorio académico que documenta un flujo completo de analítica sobre el precio de venta de viviendas residenciales en Arizona (Estados Unidos). El trabajo se organiza en tres etapas secuenciales: evaluación de calidad de datos, construcción y evaluación del modelo de minería de datos, y despliegue de una interfaz de predicción.

## Secuencia del proyecto

El orden de ejecución y de presentación del contenido es el siguiente.

1. **Calidad de datos**  
   En el notebook `notebooks/practica_parte_calidad.ipynb` se explora el conjunto de datos, se perfilan variables y se documentan criterios de limpieza y preparación. Esta etapa fundamenta decisiones posteriores sobre variables predictoras, tratamiento de valores atípicos o faltantes y transformaciones coherentes con el objetivo de modelado.

2. **Minería de datos**  
   El notebook `notebooks/practica_parte_mineria.ipynb` continúa con el entrenamiento y la comparación de modelos de regresión para predecir `lastSoldPrice`. Incluye la definición del problema, partición de datos, ingeniería de características alineada con la fase de calidad (por ejemplo, codificación del código postal) y métricas de desempeño.

3. **Despliegue**  
   La aplicación `app.py` (Streamlit) carga los artefactos entrenados (`modelo.pkl`, `zip_encoder.pkl`) y expone una interfaz para introducir atributos de una vivienda y obtener una estimación de precio, replicando la preparación de datos utilizada en entrenamiento.

## Contexto del modelo

- **Objetivo:** regresión del precio de venta (`lastSoldPrice`) de propiedades residenciales en Arizona.  
- **Origen de los datos:** conjunto público en Kaggle denominado *Arizona Real Estate: Sold Properties Dataset 2026*, con transacciones residenciales cerradas y variables estructurales (superficie, habitaciones, baños, garaje, ubicación mediante código postal, entre otras).  
- **Variables de entrada en la aplicación:** `year_built`, `sqft`, `stories`, `beds`, `baths`, `baths_full`, `garage` y `zip` (codificado de forma consistente con el pipeline de calidad y minería).

## Integrantes

- Alfonso Gabriel Jiménez Rivas  
- Sebastián Soto Ángel  

## Estructura del repositorio

| Ruta | Descripción |
|------|-------------|
| `notebooks/practica_parte_calidad.ipynb` | Análisis y reporte de calidad de datos |
| `notebooks/practica_parte_mineria.ipynb` | Modelado, evaluación y exportación del modelo |
| `app.py` | Aplicación Streamlit para inferencia |
| `modelo.pkl` | Modelo serializado (Random Forest y escalador asociado) |
| `zip_encoder.pkl` | Codificador del código postal compatible con el entrenamiento |
| `requirements.txt` | Dependencias de Python para `app.py` |

Los notebooks leen y escriben artefactos en la **raíz del repositorio** (por ejemplo `datos_limpios_arizona.csv`, `zip_encoder.pkl`, `modelo.pkl`) mediante rutas relativas `../…`, de modo que la aplicación en raíz sigue encontrando los mismos ficheros.

Si utilizas el CSV crudo de Kaggle, colócalo en la raíz del proyecto con el nombre `arizona_sold_properties_2026.csv` antes de ejecutar el notebook de calidad.

## Entorno y ejecución local

Se recomienda Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La aplicación localiza los ficheros `modelo.pkl` y `zip_encoder.pkl` en el mismo directorio que `app.py`.

## Publicación en Streamlit Community Cloud

Configuración habitual para desplegar este repositorio:

1. Conectar el repositorio de GitHub en [Streamlit Community Cloud](https://streamlit.io/cloud).  
2. **Archivo principal:** `app.py` (indícalo en la configuración avanzada si la plataforma no lo detecta automáticamente).  
3. **Dependencias:** archivo `requirements.txt` en la raíz del repositorio.  
4. No es obligatorio definir secretos para esta aplicación; los artefactos deben estar versionados en el repositorio o disponibles mediante el método de despliegue que elijas.

Tras el despliegue, comprueba en los registros de la aplicación que las versiones instaladas coincidan razonablemente con las usadas al serializar el modelo (especialmente `scikit-learn`), para evitar advertencias o errores al cargar los objetos `pickle`.

## Limitaciones conocidas

La interfaz muestra una advertencia sobre el error porcentual medio absoluto (MAPE) del modelo en torno al 22,5 %. Las predicciones deben interpretarse como orientativas y no como tasaciones oficiales.

## Licencia y uso académico

Este material se presenta en el marco de una práctica de asignatura. Revisa las licencias del dataset en Kaggle antes de un uso distinto al educativo.
