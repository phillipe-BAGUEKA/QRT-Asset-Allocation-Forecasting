"""Entry point for the QRT Streamlit multipage application."""

from __future__ import annotations

import logging

import streamlit as st

from app.config import QRT_LOGO_PATH
from frontend.views import api_overview, home, model_info, prediction


LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Configure and run the Streamlit multipage navigation."""
    st.set_page_config(
        page_title="QRT Inference",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if QRT_LOGO_PATH.is_file():
        try:
            st.logo(str(QRT_LOGO_PATH), size="large")
        except Exception:
            LOGGER.exception("Streamlit could not display the QRT logo.")
    else:
        LOGGER.error("QRT logo not found at %s", QRT_LOGO_PATH)

    pages = {
        "QRT Inference": [
            st.Page(
                home.render,
                title="Accueil",
                icon="🏠",
                url_path="home",
                default=True,
            ),
            st.Page(
                prediction.render,
                title="Prédiction",
                icon="📈",
                url_path="prediction",
            ),
            st.Page(
                model_info.render,
                title="Modèle",
                icon="🧠",
                url_path="model",
            ),
            st.Page(
                api_overview.render,
                title="API",
                icon="🔌",
                url_path="api",
            ),
        ]
    }
    navigation = st.navigation(pages, position="sidebar", expanded=True)
    navigation.run()


if __name__ == "__main__":
    main()
