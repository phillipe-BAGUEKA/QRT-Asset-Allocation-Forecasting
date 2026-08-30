'''Professional Streamlit demonstration backed exclusively by FastAPI.'''

from __future__ import annotations

import streamlit as st

from frontend.api_client import ApiClientError, get_health, get_model_info, predict


def main() -> None:
    '''Render one focused inference page for the final portfolio model.'''
    st.set_page_config(
        page_title='QRT Return Direction',
        page_icon=':chart_with_upwards_trend:',
        layout='wide',
    )
    st.title('QRT Asset-Allocation Return Direction')
    st.caption('A reproducible educational machine-learning demonstration')
    st.write(
        'Enter the twenty raw lagged returns. The interface sends the request '
        'to FastAPI; it never loads or trains the model directly.'
    )

    try:
        health = get_health()
        model_info = get_model_info()
    except ApiClientError as error:
        st.error(f'FastAPI is unavailable: {error}')
        st.info('Start the API, then refresh this page.')
        st.stop()

    if not health.get('model_loaded') or not health.get('schema_available'):
        st.error('The API is running but the final model is not ready.')
        st.stop()

    st.success(
        f"API ready - {model_info.get('model_name')} "
        f"v{model_info.get('model_version')}"
    )
    features = model_info.get('features', [])
    if len(features) != 20:
        st.error('FastAPI returned an invalid feature schema.')
        st.stop()

    values: dict[str, float | None] = {}
    with st.form('prediction'):
        columns = st.columns(4)
        for index, feature in enumerate(features):
            with columns[index % 4]:
                value = st.number_input(
                    feature,
                    value=0.0,
                    format='%.8f',
                    help='Raw lagged return; missing values are imputed by the pipeline.',
                )
                values[feature] = float(value)
        submitted = st.form_submit_button('Predict', type='primary')

    if submitted:
        try:
            result = predict(values)
        except ApiClientError as error:
            st.error(f'Prediction failed: {error}')
        else:
            metrics = st.columns(3)
            metrics[0].metric('Predicted class', result['predicted_class'])
            metrics[1].metric(
                'Positive probability', f"{result['positive_probability']:.2%}"
            )
            metrics[2].metric('Decision threshold', result['threshold'])

    st.divider()
    st.warning(
        'Educational portfolio project only. This model is not financial '
        'advice and provides no guarantee of investment performance.'
    )


if __name__ == '__main__':
    main()
