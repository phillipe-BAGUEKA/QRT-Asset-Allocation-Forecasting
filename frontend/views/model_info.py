"""Model metadata page backed exclusively by the API."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.api_client import ApiClientError, get_model_info


NOT_AVAILABLE = "Information non disponible"


def _display_value(value: Any) -> Any:
    return NOT_AVAILABLE if value is None else value


def render() -> None:
    """Render model identity, training information, and diagnostics."""
    st.title("Informations du modèle")
    try:
        model_info = get_model_info()
    except ApiClientError as exc:
        st.error(f"Informations indisponibles : {exc}")
        return

    identity_columns = st.columns(4)
    identity_columns[0].metric(
        "Nom", _display_value(model_info.get("model_name"))
    )
    identity_columns[1].metric(
        "Version", _display_value(model_info.get("model_version"))
    )
    identity_columns[2].metric(
        "Type", _display_value(model_info.get("model_type"))
    )
    identity_columns[3].metric(
        "Seuil", _display_value(model_info.get("threshold"))
    )

    training_columns = st.columns(2)
    training_columns[0].metric(
        "Observations",
        _display_value(model_info.get("training_observations")),
    )
    training_columns[1].metric(
        "Dates", _display_value(model_info.get("training_dates"))
    )

    period = model_info.get("training_period")
    if isinstance(period, dict):
        st.write(
            "**Période d'entraînement :** "
            f"{period.get('start') or '?'} → {period.get('end') or '?'}"
        )
    else:
        st.write(f"**Période d'entraînement :** {NOT_AVAILABLE}")

    pipeline_steps = model_info.get("pipeline_steps")
    st.subheader("Pipeline")
    if isinstance(pipeline_steps, list):
        st.code(" → ".join(str(step) for step in pipeline_steps))
    else:
        st.write(NOT_AVAILABLE)

    st.subheader("Features et ordre")
    features = model_info.get("features")
    if isinstance(features, list):
        st.dataframe(
            [
                {"Position": index, "Feature": feature}
                for index, feature in enumerate(features, start=1)
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.write(NOT_AVAILABLE)

    with st.expander("Métriques de référence", expanded=True):
        metrics = model_info.get("reference_metrics")
        st.json(metrics) if isinstance(metrics, dict) else st.write(
            NOT_AVAILABLE
        )

    with st.expander("Versions logicielles"):
        versions = model_info.get("software_versions")
        if isinstance(versions, dict):
            st.dataframe(
                [
                    {"Composant": component, "Version": version}
                    for component, version in versions.items()
                ],
                hide_index=True,
                width="stretch",
            )
        else:
            st.write(NOT_AVAILABLE)

    st.subheader("Empreinte du modèle")
    model_hash = model_info.get("model_sha256")
    st.code(model_hash) if model_hash else st.write(NOT_AVAILABLE)
