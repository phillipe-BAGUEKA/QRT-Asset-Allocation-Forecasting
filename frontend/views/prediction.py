"""Interactive single-allocation prediction page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.api_client import (
    ApiClientError,
    ApiResponseError,
    ApiTimeoutError,
    get_model_info,
    predict,
)


def _show_prediction_error(error: ApiClientError) -> None:
    if isinstance(error, ApiTimeoutError):
        st.error("L'API n'a pas répondu dans le délai imparti.")
    elif isinstance(error, ApiResponseError) and error.status_code == 422:
        st.error("Les valeurs envoyées ne respectent pas le schéma attendu.")
    elif isinstance(error, ApiResponseError) and error.status_code == 503:
        st.error("Le modèle ou ses métadonnées sont indisponibles.")
    elif isinstance(error, ApiResponseError) and error.status_code >= 500:
        st.error("L'API a rencontré une erreur interne.")
    else:
        st.error(f"La prédiction est indisponible : {error}")


def _render_result(result: dict[str, Any]) -> None:
    probability = float(result["positive_probability"])
    predicted_class = int(result["predicted_class"])
    percentage = f"{probability * 100:.2f} %".replace(".", ",")

    st.success("Prédiction calculée par le pipeline QRT")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Probabilité positive", percentage)
    metric_columns[1].metric("Classe prédite", predicted_class)
    metric_columns[2].metric("Seuil", result["threshold"])

    st.write(
        f"**Modèle :** {result['model_name']}  \n"
        f"**Version :** {result.get('model_version') or 'Information non disponible'}"
    )
    if predicted_class == 1:
        st.info("Le modèle prédit une performance future positive.")
    else:
        st.info("Le modèle ne prédit pas une performance future positive.")
    st.caption(
        "La probabilité est une estimation du modèle, pas une certitude ni "
        "un conseil financier."
    )


def render() -> None:
    """Render the live prediction form backed by FastAPI."""
    st.title("Prédiction individuelle")
    st.write(
        "Renseignez les rendements historiques. Une valeur vide est envoyée "
        "comme `null` et reste traitée par l'imputer du pipeline."
    )

    try:
        model_info = get_model_info()
    except ApiClientError as exc:
        st.error(f"Impossible de récupérer le schéma du modèle : {exc}")
        return

    features = model_info.get("features")
    if not isinstance(features, list) or not all(
        isinstance(feature, str) for feature in features
    ):
        st.error("L'API n'a pas retourné une liste de features valide.")
        return

    if st.button("Réinitialiser le formulaire"):
        for feature in features:
            st.session_state.pop(f"prediction_{feature}", None)
        st.rerun()

    values: dict[str, float | None] = {}
    with st.form("qrt_prediction_form"):
        form_columns = st.columns(2)
        for index, feature in enumerate(features):
            with form_columns[index % 2]:
                value = st.number_input(
                    feature,
                    value=None,
                    format="%.8f",
                    placeholder="Valeur vide = donnée manquante",
                    help="Rendement historique utilisé par le modèle QRT.",
                    key=f"prediction_{feature}",
                )
                values[feature] = (
                    None if value is None else float(value)
                )
        submitted = st.form_submit_button("Prédire", type="primary")

    if not submitted:
        return

    with st.spinner("Interrogation de l'API FastAPI..."):
        try:
            result = predict(values)
        except ApiClientError as exc:
            _show_prediction_error(exc)
            return

    _render_result(result)
    st.subheader("Valeurs envoyées")
    st.dataframe(
        [
            {
                "Feature": feature,
                "Valeur": (
                    "Manquante (null)"
                    if value is None
                    else f"{value:.8g}"
                ),
            }
            for feature, value in values.items()
        ],
        hide_index=True,
        width="stretch",
    )
