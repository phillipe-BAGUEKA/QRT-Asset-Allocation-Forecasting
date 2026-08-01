"""Tests for the reproducible final Gradient Boosting training runner."""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import train_final_gradient_boosting as training


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_raw_training_data(
    row_count: int = 48,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_ids = np.arange(row_count)
    X_train = pd.DataFrame(
        {
            "ROW_ID": row_ids,
            "TS": [f"DATE_{index // 8 + 1:04d}" for index in row_ids],
            **{
                f"RET_{feature_index}": (
                    np.sin(row_ids + feature_index) / 100.0
                )
                for feature_index in range(1, 21)
            },
            "UNUSED_FEATURE": np.ones(row_count),
        }
    )
    y_train = pd.DataFrame(
        {
            "ROW_ID": row_ids,
            "target": np.where(row_ids % 2 == 0, -0.01, 0.01),
        }
    )
    return X_train, y_train


def _build_joined_training_data(row_count: int = 48) -> pd.DataFrame:
    X_train, y_train = _build_raw_training_data(row_count)
    joined = X_train.merge(y_train, on="ROW_ID", how="inner")
    joined[training.TARGET_COLUMN] = (
        joined["target"] > 0
    ).astype(int)
    return joined


def _fit_small_final_model() -> tuple[object, pd.DataFrame]:
    joined = _build_joined_training_data()
    X, y = training._prepare_training_matrices(joined)
    model = training._build_final_model()
    model.fit(X, y)
    return model, X


def test_import_has_no_training_or_real_model_side_effects() -> None:
    models_directory = PROJECT_ROOT / "models"
    files_before = (
        sorted(path.relative_to(models_directory) for path in models_directory.rglob("*"))
        if models_directory.exists()
        else []
    )

    imported_module = importlib.reload(training)

    files_after = (
        sorted(path.relative_to(models_directory) for path in models_directory.rglob("*"))
        if models_directory.exists()
        else []
    )
    assert callable(imported_module.main)
    assert files_after == files_before


def test_constants_and_feature_order_are_frozen() -> None:
    assert training.MODEL_NAME == "gradient_boosting_ret20_v1"
    assert training.TARGET_COLUMN == "class"
    assert training.PREDICTION_THRESHOLD == 0.50
    assert training.RANDOM_STATE == 42
    assert training.FEATURE_COLUMNS == [
        f"RET_{index}" for index in range(1, 21)
    ]
    assert training.FINAL_MODEL_PARAMS == {
        "learning_rate": 0.05,
        "n_estimators": 50,
        "max_depth": 2,
        "min_samples_leaf": 20,
        "subsample": 0.7,
        "max_features": "sqrt",
        "random_state": 42,
    }


def test_final_model_uses_exact_pipeline_and_parameters() -> None:
    model = training._build_final_model()
    classifier = model.named_steps["classifier"]

    assert type(model.named_steps["imputer"]).__name__ == "SimpleImputer"
    assert type(classifier).__name__ == "GradientBoostingClassifier"
    assert {
        key: classifier.get_params()[key]
        for key in training.FINAL_MODEL_PARAMS
    } == training.FINAL_MODEL_PARAMS


@pytest.mark.parametrize(
    ("frame_name", "missing_column"),
    [
        ("X_train", "ROW_ID"),
        ("X_train", "TS"),
        ("X_train", "RET_1"),
        ("X_train", "RET_20"),
        ("y_train", "ROW_ID"),
        ("y_train", "target"),
    ],
)
def test_required_columns_are_validated(
    frame_name: str,
    missing_column: str,
) -> None:
    X_train, y_train = _build_raw_training_data()
    frame = X_train if frame_name == "X_train" else y_train
    frame.drop(columns=missing_column, inplace=True)

    with pytest.raises(ValueError, match="missing required columns"):
        training._validate_raw_training_data(X_train, y_train)


@pytest.mark.parametrize("frame_name", ["X_train", "y_train"])
def test_missing_row_ids_are_rejected(frame_name: str) -> None:
    X_train, y_train = _build_raw_training_data()
    frame = X_train if frame_name == "X_train" else y_train
    frame.loc[0, "ROW_ID"] = np.nan

    with pytest.raises(ValueError, match="missing ROW_ID"):
        training._validate_raw_training_data(X_train, y_train)


@pytest.mark.parametrize("frame_name", ["X_train", "y_train"])
def test_duplicate_row_ids_are_rejected(frame_name: str) -> None:
    X_train, y_train = _build_raw_training_data()
    frame = X_train if frame_name == "X_train" else y_train
    frame.loc[1, "ROW_ID"] = frame.loc[0, "ROW_ID"]

    with pytest.raises(ValueError, match="duplicate ROW_ID"):
        training._validate_raw_training_data(X_train, y_train)


def test_different_row_id_sets_are_rejected() -> None:
    X_train, y_train = _build_raw_training_data()
    y_train.loc[0, "ROW_ID"] = int(y_train["ROW_ID"].max()) + 1

    with pytest.raises(ValueError, match="exactly the same ROW_ID"):
        training._validate_raw_training_data(X_train, y_train)


@pytest.mark.parametrize(
    ("invalid_target", "expected_message"),
    [
        ([0.1, np.nan], "missing values"),
        ([0.1, np.inf], "finite values"),
        ([0.1, -np.inf], "finite values"),
        (["negative", "positive"], "numeric dtype"),
        ([False, True], "boolean dtype"),
    ],
)
def test_invalid_raw_target_is_rejected_before_class_creation(
    invalid_target: list[object],
    expected_message: str,
) -> None:
    X_train, y_train = _build_raw_training_data()
    repeated_target = np.resize(invalid_target, len(y_train))
    y_train["target"] = repeated_target

    with pytest.raises(ValueError, match=expected_message):
        training._validate_raw_training_data(X_train, y_train)

    assert "class" not in y_train.columns


@pytest.mark.parametrize(
    "classes",
    [
        np.zeros(48, dtype=int),
        np.where(np.arange(48) % 2 == 0, 0, 2),
        np.where(np.arange(48) == 0, np.nan, 1.0),
    ],
)
def test_non_binary_or_missing_target_is_rejected(
    classes: np.ndarray,
) -> None:
    joined = _build_joined_training_data()
    joined[training.TARGET_COLUMN] = classes

    with pytest.raises(ValueError, match="class"):
        training._validate_joined_training_data(joined, len(joined))


def test_join_row_count_change_is_rejected() -> None:
    joined = _build_joined_training_data()

    with pytest.raises(ValueError, match="changed the number of rows"):
        training._validate_joined_training_data(
            joined,
            len(joined) + 1,
        )


def test_only_ordered_ret_features_are_prepared() -> None:
    joined = _build_joined_training_data()
    X, y = training._prepare_training_matrices(joined)

    assert X.columns.tolist() == training.FEATURE_COLUMNS
    assert "UNUSED_FEATURE" not in X.columns
    assert y.name == training.TARGET_COLUMN


def test_joblib_round_trip_json_and_sha256_use_tmp_path(
    tmp_path: Path,
) -> None:
    real_models_directory = PROJECT_ROOT / "models"
    real_files_before = (
        sorted(real_models_directory.rglob("*"))
        if real_models_directory.exists()
        else []
    )
    model, X = _fit_small_final_model()
    sample = training._create_sanity_sample(X)
    probabilities_before, predictions_before = (
        training._predict_and_validate(model, sample)
    )
    model_path = tmp_path / "models" / f"{training.MODEL_NAME}.joblib"
    metadata_path = (
        tmp_path / "models" / f"{training.MODEL_NAME}.metadata.json"
    )

    artifact_size = training._atomic_dump_model(model, model_path)
    reloaded_model = training._verify_model_round_trip(
        model_path,
        sample,
        probabilities_before,
        predictions_before,
    )
    probabilities_after, predictions_after = (
        training._predict_and_validate(reloaded_model, sample)
    )
    artifact_sha256 = training._sha256_file(model_path)
    expected_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = training._build_metadata(
        git_commit="a" * 40,
        git_branch="test-branch",
        training_row_count=len(X),
        training_date_count=6,
        training_start_date="DATE_0001",
        training_end_date="DATE_0006",
        training_positive_rate=0.5,
        fit_time_seconds=0.01,
        artifact_filename=model_path.name,
        artifact_size_bytes=artifact_size,
        artifact_sha256=artifact_sha256,
    )
    training._atomic_write_json(metadata, metadata_path)
    loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    np.testing.assert_allclose(
        probabilities_before,
        probabilities_after,
        rtol=1e-12,
        atol=1e-15,
    )
    assert np.array_equal(predictions_before, predictions_after)
    assert model_path.is_file()
    assert artifact_size == model_path.stat().st_size > 0
    assert artifact_sha256 == expected_sha256
    assert loaded_metadata == metadata
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
    assert loaded_metadata["artifact_sha256"] == expected_sha256
    real_files_after = (
        sorted(real_models_directory.rglob("*"))
        if real_models_directory.exists()
        else []
    )
    assert real_files_after == real_files_before


def test_runner_has_no_x_test_optuna_or_mlflow_imports() -> None:
    source = Path(training.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert "load_X_test" not in source
    assert "load_sample_submission" not in source
    assert not any(name.startswith("optuna") for name in imported_modules)
    assert not any(name.startswith("mlflow") for name in imported_modules)
