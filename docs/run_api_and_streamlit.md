# Run the final inference application

Install the serving dependencies from the repository root:

```powershell
python -m pip install -e '.[app]'
```

Start FastAPI:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In another terminal, start Streamlit:

```powershell
$env:QRT_API_BASE_URL = 'http://localhost:8000'
python -m streamlit run frontend/streamlit_app.py
```

Streamlit always calls FastAPI. It does not load the Joblib artifact directly.
For Docker, `compose.yaml` sets the internal URL to `http://api:8000`.
