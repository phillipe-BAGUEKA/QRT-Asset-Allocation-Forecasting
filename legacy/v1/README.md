# V1 legacy archive

This directory preserves the research and modeling methodology that was
active at Git commit:

```text
e64aa6064e723bad935651bd0c664ea9c044ffc0
```

V1 interpreted the opaque `TS` identifiers as an ordered chronology and used
four expanding-window folds. That assumption is no longer part of the active
methodology. The archived source, notebooks, scripts, historical tests and
submission log are retained read-only for traceability; they are not installed
as the `qrt_forecasting` package and are not collected by the active test suite.

Exact historical reconstruction should start from the commit above. No legacy
Git tag was created during this refactor.

The active FastAPI and Streamlit application lives outside this directory and
does not import or serve any file from `legacy/v1`.

The existing notebook 09 stash is deliberately not represented here. The
archive contains only the version tracked by Git at the reference commit.
