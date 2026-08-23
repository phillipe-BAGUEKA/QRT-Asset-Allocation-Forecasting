from __future__ import annotations

import hashlib
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

from qrt_forecasting.v2.gradient_boosting import (
    REFERENCE_FEATURE_COLUMNS,
    build_gradient_boosting_reference_pipeline,
    file_sha256,
    load_experiment_configuration,
    sha256_canonical_text,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_reference_pipeline_matches_the_frozen_specification() -> None:
    pipeline = build_gradient_boosting_reference_pipeline()

    assert isinstance(pipeline.named_steps['imputer'], SimpleImputer)
    assert pipeline.named_steps['imputer'].strategy == 'constant'
    assert pipeline.named_steps['imputer'].fill_value == 0.0
    classifier = pipeline.named_steps['classifier']
    assert isinstance(classifier, GradientBoostingClassifier)
    assert classifier.learning_rate == 0.05
    assert classifier.n_estimators == 50
    assert classifier.max_depth == 2
    assert classifier.min_samples_leaf == 20
    assert classifier.subsample == 0.7
    assert classifier.max_features == 'sqrt'
    assert classifier.random_state == 42


def test_pipeline_factory_returns_fresh_unfitted_instances() -> None:
    first = build_gradient_boosting_reference_pipeline()
    second = build_gradient_boosting_reference_pipeline()

    assert first is not second
    assert first.named_steps['classifier'] is not second.named_steps['classifier']
    assert not hasattr(first.named_steps['classifier'], 'estimators_')


def test_reference_configuration_is_exact() -> None:
    configuration = load_experiment_configuration(
        REPOSITORY_ROOT / 'configs' / 'experiments' / 'GB_RET20_REFERENCE.toml'
    )

    assert configuration['feature_columns'] == REFERENCE_FEATURE_COLUMNS
    assert configuration['threshold'] == 0.5
    assert configuration['random_state'] == 42
    assert configuration['validation']['n_splits'] == 5


def test_canonical_text_hash_is_identical_for_lf_and_crlf(tmp_path: Path) -> None:
    lf_path = tmp_path / 'lf.toml'
    crlf_path = tmp_path / 'crlf.toml'
    lf_path.write_bytes(b'[model]\nlearning_rate = 0.05\n')
    crlf_path.write_bytes(b'[model]\r\nlearning_rate = 0.05\r\n')

    assert sha256_canonical_text(lf_path) == sha256_canonical_text(crlf_path)


def test_canonical_text_hash_detects_a_real_parameter_change(
    tmp_path: Path,
) -> None:
    reference_path = tmp_path / 'reference.toml'
    changed_path = tmp_path / 'changed.toml'
    reference_path.write_bytes(b'[model]\nlearning_rate = 0.05\n')
    changed_path.write_bytes(b'[model]\nlearning_rate = 0.10\n')

    assert sha256_canonical_text(reference_path) != sha256_canonical_text(
        changed_path
    )


def test_canonical_text_hash_preserves_final_newline_semantics(
    tmp_path: Path,
) -> None:
    with_newline = tmp_path / 'with_newline.toml'
    without_newline = tmp_path / 'without_newline.toml'
    with_newline.write_bytes(b'[model]\n')
    without_newline.write_bytes(b'[model]')

    assert sha256_canonical_text(with_newline) != sha256_canonical_text(
        without_newline
    )


def test_file_hash_remains_strictly_byte_for_byte(tmp_path: Path) -> None:
    lf_path = tmp_path / 'lf.bin'
    crlf_path = tmp_path / 'crlf.bin'
    binary_content = b'\x00\xff\r\n\x10'
    lf_path.write_bytes(binary_content.replace(b'\r\n', b'\n'))
    crlf_path.write_bytes(binary_content)

    assert file_sha256(crlf_path) == hashlib.sha256(binary_content).hexdigest()
    assert file_sha256(lf_path) != file_sha256(crlf_path)
