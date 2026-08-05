"""Home page for the QRT inference application."""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from app.config import QRT_LOGO_PATH
from frontend.api_client import ApiClientError, get_health, get_model_info


LOGGER = logging.getLogger(__name__)
NOT_AVAILABLE = "Information non disponible"


def _show_logo() -> None:
    if QRT_LOGO_PATH.is_file():
        st.image(str(QRT_LOGO_PATH), width=260)
    else:
        LOGGER.error("QRT logo not found at %s", QRT_LOGO_PATH)
        st.warning("Le logo QRT est indisponible.")


def _training_period(model_info: dict[str, Any]) -> str:
    period = model_info.get("training_period")
    if not isinstance(period, dict):
        return NOT_AVAILABLE
    start = period.get("start")
    end = period.get("end")
    if start is None and end is None:
        return NOT_AVAILABLE
    return f"{start or '?'} → {end or '?'}"


def render() -> None:
    """Render the educational landing page."""
    _show_logo()
    st.title("QRT Asset Allocation Forecasting")
    st.caption("Trust or Short? Predicting Daily Asset Allocation Performance")

    st.markdown(
        "Cette application prédit si la performance future d'une allocation "
        "d'actifs sera positive (`classe 1`) ou non positive (`classe 0`)."
    )
    st.info("Architecture : **Streamlit → FastAPI → Pipeline QRT**")

    try:
        health = get_health()
    except ApiClientError as exc:
        st.error(f"API indisponible : {exc}")
        st.caption(
            "Démarrez FastAPI avant d'utiliser les pages de prédiction."
        )
    else:
        if health.get("status") == "healthy":
            st.success("API connectée et opérationnelle")
        else:
            st.warning("L'API répond avec un état inattendu.")

    try:
        model_info = get_model_info()
    except ApiClientError as exc:
        model_info = {}
        st.warning(f"Informations du modèle indisponibles : {exc}")

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Modèle", model_info.get("model_name", NOT_AVAILABLE)
    )
    metric_columns[1].metric(
        "Features", model_info.get("feature_count", NOT_AVAILABLE)
    )
    metric_columns[2].metric(
        "Seuil", model_info.get("threshold", NOT_AVAILABLE)
    )
    metric_columns[3].metric(
        "Observations",
        model_info.get("training_observations", NOT_AVAILABLE),
    )

    st.markdown(f"**Période d'entraînement :** {_training_period(model_info)}")

    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Classe 1")
        st.write("Le modèle prédit une performance future positive.")
    with right_column:
        st.subheader("Classe 0")
        st.write("Le modèle ne prédit pas une performance future positive.")

    st.info(
        "Utilisez la navigation pour ouvrir la page **Prédiction** et "
        "interroger le pipeline via l'API."
    )
    st.warning(
        "Cette application est une démonstration de Machine Learning et ne "
        "constitue pas un conseil financier."
    )
