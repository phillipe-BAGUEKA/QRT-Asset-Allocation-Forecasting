from __future__ import annotations

import json
from pathlib import Path

from qrt_forecasting.v2.gradient_boosting import (
    load_experiment_configuration,
    sha256_canonical_text,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_gb_ret20_reference_configuration_and_report_are_immutable() -> None:
    configuration_path = (
        REPOSITORY_ROOT
        / 'configs'
        / 'experiments'
        / 'GB_RET20_REFERENCE.toml'
    )
    report_path = (
        REPOSITORY_ROOT
        / 'reports'
        / 'experiments'
        / 'GB_RET20_REFERENCE.json'
    )
    configuration = load_experiment_configuration(configuration_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))

    assert sha256_canonical_text(configuration_path) == (
        '3ef7899676d8a51ee0372b7dc592312f20fba2acfa42906eec9feea8729809df'
    )
    assert configuration['experiment_id'] == 'GB_RET20_REFERENCE'
    assert report['global_metrics']['accuracy'] == 0.5200045534964687
    assert report['oof_sha256'] == (
        'a215c10365c19e6436208d1ee49f8aaad8f2a5352d7009911a9c506438b5e3d0'
    )
    assert report['lockbox_metrics_computed'] is False
