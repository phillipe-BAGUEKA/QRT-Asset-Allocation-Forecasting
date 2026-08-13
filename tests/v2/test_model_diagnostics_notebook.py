from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _notebook_code(name: str) -> str:
    notebook = json.loads(
        (
            REPOSITORY_ROOT / 'research' / 'v2' / 'notebooks' / name
        ).read_text(encoding='utf-8')
    )
    return '\n'.join(
        ''.join(cell.get('source', []))
        for cell in notebook['cells']
        if cell['cell_type'] == 'code'
    )


def test_model_diagnostics_notebook_is_artifact_only() -> None:
    notebook_path = (
        REPOSITORY_ROOT
        / 'research'
        / 'v2'
        / 'notebooks'
        / '01_model_diagnostics.ipynb'
    )
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    code = _notebook_code('01_model_diagnostics.ipynb')

    assert 'oof_predictions.csv' in code
    assert 'fold_metrics.csv' in code
    assert 'group_metrics.csv' in code
    assert 'summary.json' in code
    assert '.fit(' not in code
    assert 'GradientBoostingClassifier' not in code
    assert 'build_gradient_boosting_reference_pipeline' not in code
    assert 'X_test' not in code
    assert 'load_training' not in code
    assert all(not cell.get('outputs') for cell in notebook['cells'])
    assert all(cell.get('execution_count') is None for cell in notebook['cells'] if cell['cell_type'] == 'code')


def test_active_v2_notebook_00_has_french_pedagogical_content() -> None:
    notebook = json.loads(
        (
            REPOSITORY_ROOT
            / 'research'
            / 'v2'
            / 'notebooks'
            / '00_grouped_validation_audit.ipynb'
        ).read_text(encoding='utf-8')
    )
    text = '\n'.join(
        ''.join(cell.get('source', [])) for cell in notebook['cells']
    )

    assert 'Audit de la validation groupée' in text
    assert 'Conclusion de l\'audit' in text
    assert 'This notebook' not in text
    assert 'Rows per fold' not in text
    assert 'Audit conclusion' not in text
