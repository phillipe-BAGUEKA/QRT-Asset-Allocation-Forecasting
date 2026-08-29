'''Build the frozen development-only V2 feature cache.'''

from __future__ import annotations

import json
from pathlib import Path

from qrt_forecasting.common.data import load_training_features
from qrt_forecasting.v2.feature_cache import build_development_feature_cache
from qrt_forecasting.v2.folds import load_assignment_artifacts


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENT_PATH = (
    REPOSITORY_ROOT / 'artifacts' / 'folds' / 'v2_grouped_assignment.csv'
)
FOLD_MANIFEST_PATH = (
    REPOSITORY_ROOT / 'reports' / 'validation'
    / 'v2_grouped_folds_manifest.json'
)
OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT / 'artifacts' / 'features' / 'v2_development'
)
RAW_FEATURE_PATH = REPOSITORY_ROOT / 'data' / 'raw' / 'X_train.csv'
EXPECTED_DEVELOPMENT_ROWS = 421_654


def main() -> None:
    assignment, _ = load_assignment_artifacts(
        ASSIGNMENT_PATH, FOLD_MANIFEST_PATH
    )
    # This loader is intentionally training-only; X_test is never resolved.
    raw_features = load_training_features()
    manifest = build_development_feature_cache(
        raw_features,
        assignment,
        output_directory=OUTPUT_DIRECTORY,
        source_path=RAW_FEATURE_PATH,
        expected_rows=EXPECTED_DEVELOPMENT_ROWS,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
