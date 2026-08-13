'''Run the frozen development-only OOF Gradient Boosting reference twice.'''

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrt_forecasting.common.data import load_training_features, load_training_target
from qrt_forecasting.common.target import build_binary_training_frame
from qrt_forecasting.v2.evaluation import evaluate_grouped_oof, evaluate_local_gate
from qrt_forecasting.v2.folds import (
    ASSIGNMENT_COLUMNS,
    DEVELOPMENT_ROLE,
    LOCKBOX_ROLE,
    canonical_dataframe_sha256,
    load_assignment_artifacts,
)
from qrt_forecasting.v2.gradient_boosting import (
    build_gradient_boosting_reference_pipeline,
    file_sha256,
    load_experiment_configuration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION_PATH = (
    REPOSITORY_ROOT / 'configs' / 'experiments' / 'GB_RET20_REFERENCE.toml'
)
ARTIFACT_DIRECTORY = (
    REPOSITORY_ROOT / 'artifacts' / 'experiments' / 'GB_RET20_REFERENCE'
)
TRACKED_REPORT_PATH = (
    REPOSITORY_ROOT / 'reports' / 'experiments' / 'GB_RET20_REFERENCE.json'
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode='w', encoding='utf-8', newline='\n', suffix='.json',
        dir=path.parent, delete=False
    ) as temporary:
        json.dump(_json_ready(payload), temporary, indent=2, sort_keys=True)
        temporary.write('\n')
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _validate_frozen_artifacts(
    configuration: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validation = configuration['validation']
    assignment_path = REPOSITORY_ROOT / validation['assignment_path']
    manifest_path = REPOSITORY_ROOT / validation['manifest_path']
    if file_sha256(manifest_path) != validation['manifest_file_sha256']:
        raise ValueError('Frozen validation manifest file hash changed.')
    assignment, manifest = load_assignment_artifacts(assignment_path, manifest_path)
    for key in (
        'data_fingerprint_sha256',
        'assignment_sha256',
        'development_identifiers_sha256',
        'lockbox_identifiers_sha256',
    ):
        if manifest.get(key) != validation[key]:
            raise ValueError(f'Frozen validation manifest field {key} changed.')
    if manifest['splitter_selection']['selected_splitter'] != 'StratifiedGroupKFold':
        raise ValueError('Unexpected frozen development splitter.')
    if canonical_dataframe_sha256(
        assignment[assignment['role'] == DEVELOPMENT_ROLE],
        columns=['ROW_ID', 'TS', 'role'],
    ) != validation['development_identifiers_sha256']:
        raise ValueError('Development identifier hash changed.')
    if canonical_dataframe_sha256(
        assignment[assignment['role'] == LOCKBOX_ROLE],
        columns=['ROW_ID', 'TS', 'role'],
    ) != validation['lockbox_identifiers_sha256']:
        raise ValueError('Lockbox identifier hash changed.')
    observed_assignment_hash = canonical_dataframe_sha256(
        assignment, columns=ASSIGNMENT_COLUMNS
    )
    if observed_assignment_hash != validation['assignment_sha256']:
        raise ValueError('Frozen assignment hash changed.')
    return assignment, manifest


def _load_development_data(
    assignment: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    development_ids = set(
        assignment.loc[assignment['role'] == DEVELOPMENT_ROLE, 'ROW_ID'].tolist()
    )
    all_assignment_ids = set(assignment['ROW_ID'].tolist())
    features = load_training_features(
        usecols=['ROW_ID', 'TS', *feature_columns]
    )
    target = load_training_target(usecols=['ROW_ID', 'target'])
    if set(features['ROW_ID'].tolist()) != all_assignment_ids:
        raise ValueError('X_train ROW_ID values differ from the frozen assignment.')
    if set(target['ROW_ID'].tolist()) != all_assignment_ids:
        raise ValueError('y_train ROW_ID values differ from the frozen assignment.')
    features = features[features['ROW_ID'].isin(development_ids)].copy()
    target = target[target['ROW_ID'].isin(development_ids)].copy()
    development = build_binary_training_frame(features, target)
    if len(development) != len(development_ids):
        raise ValueError('Development row count changed during target construction.')
    return development


def _persist_results(
    *,
    first: Any,
    second: Any,
    gate: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    first.oof_predictions.to_csv(
        ARTIFACT_DIRECTORY / 'oof_predictions.csv',
        index=False,
        lineterminator='\n',
        float_format='%.17g',
    )
    first.fold_metrics.to_csv(
        ARTIFACT_DIRECTORY / 'fold_metrics.csv', index=False, lineterminator='\n'
    )
    first.group_metrics.to_csv(
        ARTIFACT_DIRECTORY / 'group_metrics.csv', index=False, lineterminator='\n'
    )
    first.probability_histogram.to_csv(
        ARTIFACT_DIRECTORY / 'probability_histogram.csv',
        index=False,
        lineterminator='\n',
    )
    report = {
        'schema_version': 1,
        'experiment_id': configuration['experiment_id'],
        'evaluation_scope': 'development_oof_only',
        'n_development_rows': int(len(first.oof_predictions)),
        'n_folds': configuration['validation']['n_splits'],
        'features': configuration['feature_columns'],
        'threshold': configuration['threshold'],
        'random_state': configuration['random_state'],
        'configuration_sha256': file_sha256(CONFIGURATION_PATH),
        'assignment_sha256': configuration['validation']['assignment_sha256'],
        'development_identifiers_sha256': configuration['validation'][
            'development_identifiers_sha256'
        ],
        'oof_sha256': first.oof_sha256,
        'global_metrics': first.global_metrics,
        'fold_metrics': first.fold_metrics.drop(
            columns=['fit_time_seconds', 'prediction_time_seconds']
        ).to_dict(orient='records'),
        'confusion_matrix': {
            'true_negative': first.global_metrics['true_negative'],
            'false_positive': first.global_metrics['false_positive'],
            'false_negative': first.global_metrics['false_negative'],
            'true_positive': first.global_metrics['true_positive'],
        },
        'timing': {
            'run_1_fit_time_seconds': first.total_fit_time_seconds,
            'run_1_prediction_time_seconds': first.total_prediction_time_seconds,
            'run_2_fit_time_seconds': second.total_fit_time_seconds,
            'run_2_prediction_time_seconds': second.total_prediction_time_seconds,
        },
        'reproducibility': {
            'run_1_oof_sha256': first.oof_sha256,
            'run_2_oof_sha256': second.oof_sha256,
            'exact_probability_match': True,
        },
        'gate': gate,
        'lockbox_metrics_computed': False,
    }
    _atomic_json_write(ARTIFACT_DIRECTORY / 'summary.json', report)
    _atomic_json_write(ARTIFACT_DIRECTORY / 'local_gate.json', gate)
    _atomic_json_write(TRACKED_REPORT_PATH, report)
    return report


def main() -> None:
    configuration = load_experiment_configuration(CONFIGURATION_PATH)
    assignment, _ = _validate_frozen_artifacts(configuration)
    development = _load_development_data(
        assignment, configuration['feature_columns']
    )
    evaluation_arguments = {
        'feature_columns': configuration['feature_columns'],
        'pipeline_factory': build_gradient_boosting_reference_pipeline,
        'threshold': configuration['threshold'],
        'n_splits': configuration['validation']['n_splits'],
    }
    first = evaluate_grouped_oof(development, assignment, **evaluation_arguments)
    second = evaluate_grouped_oof(development, assignment, **evaluation_arguments)
    gate = evaluate_local_gate(first, second)
    report = _persist_results(
        first=first,
        second=second,
        gate=gate,
        configuration=configuration,
    )
    print(json.dumps(_json_ready(report), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
