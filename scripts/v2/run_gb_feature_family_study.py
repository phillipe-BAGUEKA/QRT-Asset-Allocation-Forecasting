'''Audit and compare frozen Gradient Boosting feature families on V2 OOF.'''

from __future__ import annotations

import json
import os
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qrt_forecasting.common.data import load_training_features, load_training_target
from qrt_forecasting.common.target import build_binary_training_frame
from qrt_forecasting.v2.comparison import (
    candidate_gate,
    paired_group_bootstrap_accuracy_gain,
)
from qrt_forecasting.common.metrics import binary_classification_metrics
from qrt_forecasting.v2.evaluation import evaluate_grouped_oof
from qrt_forecasting.v2.feature_audit import (
    audit_categorical_overlap,
    audit_feature_columns,
)
from qrt_forecasting.v2.feature_families import (
    NUMERIC_FEATURE_SETS,
    FeatureSet,
    build_feature_family_pipeline,
    categorical_feature_sets,
)
from qrt_forecasting.v2.folds import DEVELOPMENT_ROLE, load_assignment_artifacts


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION_PATH = (
    REPOSITORY_ROOT / 'configs' / 'experiments' / 'GB_FEATURE_FAMILY_STUDY.toml'
)
REFERENCE_DIRECTORY = (
    REPOSITORY_ROOT / 'artifacts' / 'experiments' / 'GB_RET20_REFERENCE'
)
STUDY_DIRECTORY = (
    REPOSITORY_ROOT / 'artifacts' / 'experiments' / 'GB_FEATURE_FAMILY_STUDY'
)
TRACKED_REPORT_PATH = (
    REPOSITORY_ROOT / 'reports' / 'experiments' / 'GB_FEATURE_FAMILY_STUDY.json'
)
ASSIGNMENT_PATH = REPOSITORY_ROOT / 'artifacts' / 'folds' / 'v2_grouped_assignment.csv'
FOLD_MANIFEST_PATH = (
    REPOSITORY_ROOT / 'reports' / 'validation' / 'v2_grouped_folds_manifest.json'
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def _load_configuration() -> dict[str, Any]:
    with CONFIGURATION_PATH.open('rb') as source:
        return tomllib.load(source)


def _load_inputs(
    configuration: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    assignment, fold_manifest = load_assignment_artifacts(
        ASSIGNMENT_PATH, FOLD_MANIFEST_PATH
    )
    validation = configuration['validation']
    if fold_manifest['assignment_sha256'] != validation['assignment_sha256']:
        raise ValueError('Frozen assignment hash changed.')
    if (
        fold_manifest['development_identifiers_sha256']
        != validation['development_identifiers_sha256']
    ):
        raise ValueError('Frozen development identifier hash changed.')
    raw_train = load_training_features()
    raw_test = pd.read_csv(REPOSITORY_ROOT / 'data' / 'raw' / 'X_test.csv')
    development_ids = set(
        assignment.loc[
            assignment['role'] == DEVELOPMENT_ROLE, 'ROW_ID'
        ].tolist()
    )
    development_features = raw_train[
        raw_train['ROW_ID'].isin(development_ids)
    ].copy()
    target = load_training_target(usecols=['ROW_ID', 'target'])
    development_target = target[target['ROW_ID'].isin(development_ids)].copy()
    development = build_binary_training_frame(
        development_features, development_target
    )
    if len(development) != 421654:
        raise ValueError('Unexpected development row count.')
    del raw_train, target, development_target
    return development, raw_test, assignment, fold_manifest


def _persist_feature_audit(
    development: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]], bool]:
    raw_development = development.drop(columns=['target', 'class'])
    inventory = audit_feature_columns(raw_development, test_features)
    categorical = audit_categorical_overlap(raw_development, test_features)
    combined_cardinality = sum(
        item['development_cardinality'] for item in categorical
    )
    combined_reasonable = (
        all(item['one_hot_fold_safe_reasonable'] for item in categorical)
        and combined_cardinality <= 500
    )
    STUDY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(
        STUDY_DIRECTORY / 'feature_inventory.csv', index=False, lineterminator='\n'
    )
    audit_payload = {
        'scope': 'development_features_and_test_schema_without_test_target',
        'excluded_features': {
            'ROW_ID': 'identifiant sans signification prédictive autorisée',
            'TS': 'groupe de validation réservé au découpage sans fuite',
            'TARGET': 'cible et non variable explicative',
        },
        'categorical_overlap': categorical,
        'combined_one_hot_reasonable': combined_reasonable,
        'arbitrary_ordinal_encoding_allowed': False,
        'non_cross_fitted_target_encoding_allowed': False,
    }
    _atomic_json_write(STUDY_DIRECTORY / 'feature_audit.json', audit_payload)
    return inventory, categorical, combined_reasonable


def _run_experiment(
    feature_set: FeatureSet,
    development: pd.DataFrame,
    assignment: pd.DataFrame,
    reference_oof: pd.DataFrame,
    reference_summary: dict[str, Any],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    factory = lambda: build_feature_family_pipeline(feature_set)
    arguments = {
        'feature_columns': feature_set.input_features,
        'pipeline_factory': factory,
        'threshold': configuration['threshold'],
        'n_splits': configuration['validation']['n_splits'],
        'compute_group_metrics': False,
    }
    first = evaluate_grouped_oof(development, assignment, **arguments)
    technical_errors = []
    if first.oof_predictions['y_pred'].nunique() < 2:
        technical_errors.append('OOF class predictions are constant.')
    if first.oof_predictions['y_proba'].nunique() < 2:
        technical_errors.append('OOF probabilities are constant.')
    technical_gate = {
        'technical_status': 'FAIL' if technical_errors else 'PASS',
        'technical_errors': technical_errors,
    }
    bootstrap = paired_group_bootstrap_accuracy_gain(
        reference_oof,
        first.oof_predictions,
        n_iterations=configuration['bootstrap_iterations'],
        seed=configuration['bootstrap_seed'],
    )
    reference_folds = pd.DataFrame(reference_summary['fold_metrics']).set_index(
        'fold_id'
    )
    candidate_folds = first.fold_metrics.set_index('fold_id')
    fold_gains = [
        float(candidate_folds.loc[index, 'accuracy'] - reference_folds.loc[index, 'accuracy'])
        for index in range(configuration['validation']['n_splits'])
    ]
    gate = candidate_gate(
        candidate_accuracy=first.global_metrics['accuracy'],
        candidate_log_loss=first.global_metrics['log_loss'],
        reference_accuracy=reference_summary['global_metrics']['accuracy'],
        reference_log_loss=reference_summary['global_metrics']['log_loss'],
        reference_accuracy_gate=configuration['reference_accuracy_gate'],
        minimum_accuracy_gain=configuration['minimum_accuracy_gain'],
        fold_accuracy_gains=fold_gains,
        bootstrap_ci95_lower=bootstrap['ci95_lower'],
        maximum_log_loss_degradation=configuration[
            'maximum_log_loss_degradation'
        ],
        technical_status=technical_gate['technical_status'],
    )
    experiment_directory = STUDY_DIRECTORY / feature_set.experiment_id
    experiment_directory.mkdir(parents=True, exist_ok=True)
    first.oof_predictions.to_csv(
        experiment_directory / 'oof_predictions.csv',
        index=False,
        lineterminator='\n',
        float_format='%.17g',
    )
    first.fold_metrics.to_csv(
        experiment_directory / 'fold_metrics.csv', index=False, lineterminator='\n'
    )
    first.probability_histogram.to_csv(
        experiment_directory / 'probability_histogram.csv',
        index=False,
        lineterminator='\n',
    )
    result = {
        'experiment_id': feature_set.experiment_id,
        'numeric_features': list(feature_set.numeric_features),
        'categorical_features': list(feature_set.categorical_features),
        'add_missing_indicators': feature_set.add_missing_indicators,
        'n_input_features': len(feature_set.input_features),
        'global_metrics': first.global_metrics,
        'confusion_matrix': {
            key: first.global_metrics[key]
            for key in (
                'true_negative',
                'false_positive',
                'false_negative',
                'true_positive',
            )
        },
        'fold_metrics': first.fold_metrics.drop(
            columns=['fit_time_seconds', 'prediction_time_seconds']
        ).to_dict(orient='records'),
        'fold_accuracy_gains': fold_gains,
        'fold_accuracy_std': float(first.fold_metrics['accuracy'].std(ddof=0)),
        'bootstrap': bootstrap,
        'gate': gate,
        'technical_gate': technical_gate,
        'oof_sha256': first.oof_sha256,
        'reproducibility': {
            'run_1_oof_sha256': first.oof_sha256,
            'run_2_oof_sha256': None,
            'exact_probability_match': None,
            'real_second_run_performed': False,
        },
        'timing': {
            'run_1_fit_time_seconds': first.total_fit_time_seconds,
            'run_1_prediction_time_seconds': first.total_prediction_time_seconds,
            'run_2_fit_time_seconds': None,
            'run_2_prediction_time_seconds': None,
        },
        'lockbox_metrics_computed': False,
    }
    _atomic_json_write(experiment_directory / 'summary.json', result)
    print(
        json.dumps(
            {
                'experiment_id': feature_set.experiment_id,
                'accuracy': first.global_metrics['accuracy'],
                'gain': gate['accuracy_gain_vs_reference_artifact'],
                'ci95': [bootstrap['ci95_lower'], bootstrap['ci95_upper']],
                'admissible': gate['admissible'],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def _comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        metrics = result['global_metrics']
        rows.append(
            {
                'experiment_id': result['experiment_id'],
                'accuracy': metrics['accuracy'],
                'accuracy_gain': result['gate'][
                    'accuracy_gain_vs_reference_artifact'
                ],
                'bootstrap_ci95_lower': result['bootstrap']['ci95_lower'],
                'bootstrap_ci95_upper': result['bootstrap']['ci95_upper'],
                'non_negative_fold_count': result['gate'][
                    'non_negative_fold_count'
                ],
                'fold_accuracy_std': result['fold_accuracy_std'],
                'roc_auc': metrics['roc_auc'],
                'log_loss': metrics['log_loss'],
                'predicted_positive_rate': metrics['predicted_positive_rate'],
                'run_1_fit_time_seconds': result['timing'][
                    'run_1_fit_time_seconds'
                ],
                'admissible': result['gate']['admissible'],
                'oof_sha256': result['oof_sha256'],
            }
        )
    return pd.DataFrame(rows).sort_values(
        [
            'accuracy',
            'bootstrap_ci95_lower',
            'fold_accuracy_std',
            'log_loss',
            'run_1_fit_time_seconds',
        ],
        ascending=[False, False, True, True, True],
        kind='mergesort',
    ).reset_index(drop=True)


def _reference_row(reference_summary: dict[str, Any]) -> dict[str, Any]:
    metrics = reference_summary['global_metrics']
    return {
        'experiment_id': 'GB_RET20_REFERENCE',
        'accuracy': metrics['accuracy'],
        'accuracy_gain': 0.0,
        'bootstrap_ci95_lower': 0.0,
        'bootstrap_ci95_upper': 0.0,
        'non_negative_fold_count': 5,
        'fold_accuracy_std': float(
            pd.Series(
                [item['accuracy'] for item in reference_summary['fold_metrics']]
            ).std(ddof=0)
        ),
        'roc_auc': metrics['roc_auc'],
        'log_loss': metrics['log_loss'],
        'predicted_positive_rate': metrics['predicted_positive_rate'],
        'run_1_fit_time_seconds': reference_summary['timing'][
            'run_1_fit_time_seconds'
        ],
        'admissible': False,
        'oof_sha256': reference_summary['oof_sha256'],
    }


def _verify_best_reproducibility(
    result: dict[str, Any],
    feature_set: FeatureSet,
    development: pd.DataFrame,
    assignment: pd.DataFrame,
    configuration: dict[str, Any],
) -> None:
    '''Perform the second real OOF run only for the best overall candidate.'''
    second = evaluate_grouped_oof(
        development,
        assignment,
        feature_columns=feature_set.input_features,
        pipeline_factory=lambda: build_feature_family_pipeline(feature_set),
        threshold=configuration['threshold'],
        n_splits=configuration['validation']['n_splits'],
        compute_group_metrics=False,
    )
    first_oof = pd.read_csv(
        STUDY_DIRECTORY / feature_set.experiment_id / 'oof_predictions.csv'
    )
    if result['oof_sha256'] != second.oof_sha256:
        raise ValueError('Best candidate deterministic OOF hashes diverged.')
    result['reproducibility'].update(
        {
            'run_2_oof_sha256': second.oof_sha256,
            'canonical_prediction_match': True,
            'real_second_run_performed': True,
        }
    )
    result['timing'].update(
        {
            'run_2_fit_time_seconds': second.total_fit_time_seconds,
            'run_2_prediction_time_seconds': second.total_prediction_time_seconds,
        }
    )
    _atomic_json_write(
        STUDY_DIRECTORY / feature_set.experiment_id / 'summary.json', result
    )


def _load_validated_result(
    feature_set: FeatureSet,
    development: pd.DataFrame,
    assignment: pd.DataFrame,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    directory = STUDY_DIRECTORY / feature_set.experiment_id
    summary_path = directory / 'summary.json'
    oof_path = directory / 'oof_predictions.csv'
    if not summary_path.is_file() or not oof_path.is_file():
        raise ValueError(f'Missing resume artifact for {feature_set.experiment_id}.')
    with summary_path.open(encoding='utf-8') as source:
        result = json.load(source)
    oof = pd.read_csv(oof_path)
    expected_assignment = assignment[
        assignment['role'] == DEVELOPMENT_ROLE
    ][['ROW_ID', 'TS', 'fold_id']]
    observed = oof[['ROW_ID', 'TS', 'fold_id']]
    merged = expected_assignment.merge(
        observed,
        on='ROW_ID',
        suffixes=('_expected', '_observed'),
        validate='one_to_one',
    )
    if len(oof) != 421654 or oof['ROW_ID'].duplicated().any():
        raise ValueError('Resumed OOF coverage is invalid.')
    if not np.array_equal(merged['TS_expected'], merged['TS_observed']):
        raise ValueError('Resumed OOF TS values changed.')
    if not np.array_equal(merged['fold_id_expected'], merged['fold_id_observed']):
        raise ValueError('Resumed OOF fold IDs changed.')
    target_by_id = development.set_index('ROW_ID')['class']
    if not np.array_equal(
        oof['y_true'].to_numpy(dtype=int),
        oof['ROW_ID'].map(target_by_id).to_numpy(dtype=int),
    ):
        raise ValueError('Resumed OOF targets changed.')
    metrics = binary_classification_metrics(
        oof['y_true'], oof['y_proba'], threshold=configuration['threshold']
    )
    for name in ('accuracy', 'roc_auc', 'log_loss'):
        if not np.isclose(metrics[name], result['global_metrics'][name], atol=1e-15):
            raise ValueError(f'Resumed OOF metric {name} changed.')
    return result


def main() -> None:
    configuration = _load_configuration()
    development, test_features, assignment, fold_manifest = _load_inputs(
        configuration
    )
    inventory, categorical_audit, categorical_reasonable = _persist_feature_audit(
        development, test_features
    )
    with (REFERENCE_DIRECTORY / 'summary.json').open(encoding='utf-8') as source:
        reference_summary = json.load(source)
    reference_oof = pd.read_csv(REFERENCE_DIRECTORY / 'oof_predictions.csv')
    if reference_summary['experiment_id'] != 'GB_RET20_REFERENCE':
        raise ValueError('Reference experiment identity changed.')
    if reference_summary['oof_sha256'] != 'a215c10365c19e6436208d1ee49f8aaad8f2a5352d7009911a9c506438b5e3d0':
        raise ValueError('Reference OOF hash changed.')
    if not np.isclose(
        reference_summary['global_metrics']['accuracy'],
        0.5200045534964687,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError('Reference accuracy changed.')

    resume = '--resume-validated' in sys.argv[1:]
    run_or_load = (
        lambda feature_set: _load_validated_result(
            feature_set, development, assignment, configuration
        )
        if resume
        else _run_experiment(
            feature_set,
            development,
            assignment,
            reference_oof,
            reference_summary,
            configuration,
        )
    )
    results = [run_or_load(feature_set) for feature_set in NUMERIC_FEATURE_SETS]
    best_numeric_result = max(
        results, key=lambda result: result['global_metrics']['accuracy']
    )
    best_numeric = next(
        item
        for item in NUMERIC_FEATURE_SETS
        if item.experiment_id == best_numeric_result['experiment_id']
    )
    categorical_experiments: list[str] = []
    if categorical_reasonable:
        for feature_set in categorical_feature_sets(best_numeric):
            results.append(run_or_load(feature_set))
            categorical_experiments.append(feature_set.experiment_id)

    all_feature_sets = [
        *NUMERIC_FEATURE_SETS,
        *categorical_feature_sets(best_numeric),
    ]
    best_result = max(
        results, key=lambda result: result['global_metrics']['accuracy']
    )
    best_feature_set = next(
        item
        for item in all_feature_sets
        if item.experiment_id == best_result['experiment_id']
    )
    _verify_best_reproducibility(
        best_result,
        best_feature_set,
        development,
        assignment,
        configuration,
    )

    table = _comparison_table(results)
    table_with_reference = pd.concat(
        [pd.DataFrame([_reference_row(reference_summary)]), table],
        ignore_index=True,
    )
    table_with_reference.to_csv(
        STUDY_DIRECTORY / 'comparison.csv', index=False, lineterminator='\n'
    )
    admissible = table[table['admissible']]
    selected = (
        None if admissible.empty else str(admissible.iloc[0]['experiment_id'])
    )
    report = {
        'schema_version': 1,
        'study_id': configuration['study_id'],
        'evaluation_scope': 'development_oof_only',
        'n_development_rows': 421654,
        'n_folds': configuration['validation']['n_splits'],
        'assignment_sha256': fold_manifest['assignment_sha256'],
        'development_identifiers_sha256': fold_manifest[
            'development_identifiers_sha256'
        ],
        'reference': {
            'experiment_id': reference_summary['experiment_id'],
            'accuracy': reference_summary['global_metrics']['accuracy'],
            'oof_sha256': reference_summary['oof_sha256'],
            'immutable': True,
        },
        'feature_inventory_rows': int(len(inventory)),
        'categorical_audit': categorical_audit,
        'categorical_one_hot_executed': categorical_reasonable,
        'categorical_experiments': categorical_experiments,
        'best_numeric_experiment_id': best_numeric.experiment_id,
        'experiments': results,
        'ranking': table.to_dict(orient='records'),
        'selected_submission_candidate': selected,
        'second_submission_gate_passed': selected is not None,
        'lockbox_metrics_computed': False,
    }
    _atomic_json_write(STUDY_DIRECTORY / 'study_summary.json', report)
    _atomic_json_write(TRACKED_REPORT_PATH, report)
    print(
        json.dumps(
            {
                'selected_submission_candidate': selected,
                'second_submission_gate_passed': selected is not None,
                'comparison_path': str(
                    (STUDY_DIRECTORY / 'comparison.csv').resolve()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == '__main__':
    main()
