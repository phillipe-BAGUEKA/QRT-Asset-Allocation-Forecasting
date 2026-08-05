"""Technical architecture and API status page."""

from __future__ import annotations

import streamlit as st

from app.config import get_api_base_url
from frontend.api_client import ApiClientError, get_health


def render() -> None:
    """Render endpoints, links, architecture, and live health status."""
    api_base_url = get_api_base_url()
    st.title("API et architecture")
    st.write(f"**URL de base :** `{api_base_url}`")

    if st.button("Vérifier de nouveau l'API"):
        st.rerun()

    try:
        health = get_health()
    except ApiClientError as exc:
        st.error(f"API indisponible : {exc}")
    else:
        if health.get("status") == "healthy":
            st.success("GET /health : healthy")
        else:
            st.warning(f"Réponse inattendue : {health}")

    st.subheader("Endpoints")
    st.dataframe(
        [
            {
                "Méthode": "GET",
                "Route": "/",
                "Rôle": "Informations générales du service",
                "Exemple": '{"name": "QRT Prediction API", ...}',
            },
            {
                "Méthode": "GET",
                "Route": "/health",
                "Rôle": "Disponibilité du processus API",
                "Exemple": '{"status": "healthy"}',
            },
            {
                "Méthode": "GET",
                "Route": "/model-info",
                "Rôle": "Métadonnées publiques du pipeline",
                "Exemple": '{"model_name": "...", "features": [...] }',
            },
            {
                "Méthode": "POST",
                "Route": "/predict",
                "Rôle": "Prédiction individuelle",
                "Exemple": '{"positive_probability": 0.53, ...}',
            },
        ],
        hide_index=True,
        width="stretch",
    )

    st.subheader("Architecture")
    st.code(
        "Utilisateur\n"
        "    ↓\n"
        "Streamlit\n"
        "    ↓ HTTP\n"
        "FastAPI / Uvicorn\n"
        "    ↓\n"
        "Pipeline Joblib\n"
        "    ↓\n"
        "Prédiction JSON"
    )

    link_columns = st.columns(2)
    with link_columns[0]:
        st.link_button("Ouvrir Swagger", f"{api_base_url}/docs")
    with link_columns[1]:
        st.link_button("Ouvrir ReDoc", f"{api_base_url}/redoc")
