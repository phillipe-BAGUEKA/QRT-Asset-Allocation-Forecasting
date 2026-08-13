'''Audit grouped splitters and freeze the V2 development/lockbox assignment.'''

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

from qrt_forecasting.common.data import (
    load_training_features,
    load_training_target,
)
from qrt_forecasting.common.target import build_binary_training_frame
from qrt_forecasting.v2.folds import (
    build_manifest,
    create_development_lockbox_assignment,
    persist_assignment_artifacts,
)
from qrt_forecasting.v2.validation import audit_splitter, select_splitter


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIGURATION_PATH = (
    REPOSITORY_ROOT / 'configs' / 'validation' / 'v2_grouped_cv.toml'
)
ASSIGNMENT_PATH = (
    REPOSITORY_ROOT / 'artifacts' / 'folds' / 'v2_grouped_assignment.csv'
)
MANIFEST_PATH = (
    REPOSITORY_ROOT
    / 'reports'
    / 'validation'
    / 'v2_grouped_folds_manifest.json'
)


def _load_configuration(path: Path = CONFIGURATION_PATH) -> dict[str, Any]:
    with path.open('rb') as configuration_file:
        return tomllib.load(configuration_file)


def run_audit(
    *,
    configuration_path: Path = CONFIGURATION_PATH,
    assignment_path: Path = ASSIGNMENT_PATH,
    manifest_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    '''Execute the structural audit without loading test data or a model.'''
    configuration = _load_configuration(configuration_path)
    features = load_training_features(usecols=['ROW_ID', 'TS'])
    target = load_training_target(usecols=['ROW_ID', 'target'])
    labelled = build_binary_training_frame(features, target)[
        ['ROW_ID', 'TS', 'class']
    ]

    splitter_audits: list[dict[str, Any]] = []
    for splitter_name in configuration['candidate_splitters']:
        _, audit = audit_splitter(
            labelled,
            splitter_name=splitter_name,
            n_splits=configuration['audit_n_splits'],
            seed=configuration['seed'],
        )
        splitter_audits.append(audit)

    selected_audit = select_splitter(splitter_audits)
    assignment, assignment_details = create_development_lockbox_assignment(
        labelled,
        splitter_name=selected_audit['splitter'],
        seed=configuration['seed'],
        lockbox_n_splits=configuration['lockbox_n_splits'],
        lockbox_target_fraction=configuration['lockbox_target_fraction'],
        development_n_splits=configuration['development_n_splits'],
    )
    manifest = build_manifest(
        data=labelled,
        assignment=assignment,
        configuration=configuration,
        splitter_audits=splitter_audits,
        assignment_details=assignment_details,
    )
    persist_assignment_artifacts(
        assignment=assignment,
        manifest=manifest,
        assignment_path=assignment_path,
        manifest_path=manifest_path,
    )
    return manifest


def main() -> None:
    manifest = run_audit()
    summary = {
        'selected_splitter': manifest['splitter_selection'][
            'selected_splitter'
        ],
        'development': manifest['partition']['development'],
        'lockbox': manifest['partition']['lockbox'],
        'assignment_sha256': manifest['assignment_sha256'],
        'data_fingerprint_sha256': manifest['data_fingerprint_sha256'],
        'assignment_path': str(ASSIGNMENT_PATH.resolve()),
        'manifest_path': str(MANIFEST_PATH.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
