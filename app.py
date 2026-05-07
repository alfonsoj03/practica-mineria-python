# -*- coding: utf-8 -*-
"""Aplicación Streamlit: predicción del precio de venta de viviendas en Arizona.

Interfaz para capturar características de una propiedad, aplicar la misma preparación
(codificación del ZIP y escalado MinMax) que en la etapa de calidad de datos y generar
una estimación con el modelo Random Forest serializado.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent


@st.cache_resource
def load_artifacts():
    """Carga modelo, escalador y codificador de ZIP desde el directorio de la aplicación."""
    with open(BASE_DIR / "modelo.pkl", "rb") as f:
        modelo, _variables, min_max_scaler = pickle.load(f)
    with open(BASE_DIR / "zip_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return modelo, min_max_scaler, encoder


def main() -> None:
    st.title("Predicción de precio de venta — Arizona")

    modelo, min_max_scaler, encoder = load_artifacts()

    year_built = st.slider(
        "Año de construcción (year_built)",
        min_value=1850,
        max_value=2026,
        value=1990,
        step=1,
    )
    sqft = st.slider(
        "Superficie (sqft)",
        min_value=150,
        max_value=18000,
        value=300,
        step=1,
    )
    stories = st.slider("Niveles (stories)", min_value=1, max_value=4, value=1, step=1)
    beds = st.slider("Habitaciones (beds)", min_value=1, max_value=12, value=1, step=1)
    baths = st.slider("Baños (baths)", min_value=0, max_value=20, value=1, step=1)
    baths_full = st.slider("Baños completos (baths_full)", min_value=0, max_value=20, value=1, step=1)
    garage = st.slider("Plazas de garaje (garage)", min_value=0, max_value=20, value=1, step=1)

    zips_posibles = encoder.ordinal_encoder.mapping[0]["mapping"].index.tolist()
    zips_posibles = [z for z in zips_posibles if pd.notna(z) and z not in (-1, -2)]

    zip_input = int(st.selectbox("Código postal (zip)", zips_posibles))

    datos = [[year_built, sqft, stories, beds, baths, baths_full, garage, zip_input]]
    data = pd.DataFrame(
        datos,
        columns=[
            "year_built",
            "sqft",
            "stories",
            "beds",
            "baths",
            "baths_full",
            "garage",
            "zip",
        ],
    )

    data_preparada = data.copy()
    data_preparada["zip_encoded"] = encoder.transform(data_preparada[["zip"]])["zip"]
    data_preparada.drop(columns=["zip"], inplace=True)

    columnas_a_escalar = [
        "year_built",
        "sqft",
        "stories",
        "beds",
        "baths",
        "baths_full",
        "garage",
        "zip_encoded",
    ]
    data_preparada[columnas_a_escalar] = min_max_scaler.transform(
        data_preparada[columnas_a_escalar]
    )

    prediccion_numerica = modelo.predict(data_preparada)
    y_pred = float(prediccion_numerica[0])

    st.metric(
        "Precio estimado de venta (USD)",
        f"{y_pred:,.2f}",
    )

    data_display = data.copy()
    data_display["Predicción"] = y_pred
    with st.expander("Valores de entrada y predicción"):
        st.dataframe(data_display, use_container_width=True)

    st.warning(
        "El modelo tiene un error (MAPE) del 22,5 %. Por cada predicción, el precio "
        "estimado puede desviarse en promedio un 22,5 % del precio real de venta."
    )


if __name__ == "__main__":
    main()
