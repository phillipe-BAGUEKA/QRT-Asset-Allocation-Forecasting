# Methodology

The active V2 protocol treats `TS` as an opaque group identifier. It does not
infer chronological order from anonymized labels. A model-free audit compared
`GroupKFold` and `StratifiedGroupKFold` against criteria fixed in advance:
complete row coverage, no group overlap, both classes in every partition, and
balance of rows, groups, and prevalence. `StratifiedGroupKFold` was selected.

The frozen assignment contains 527,073 labelled rows across 2,522 intact `TS`
groups. Development contains 421,654 rows and 2,028 groups; the lockbox contains
105,419 rows and 494 groups. Five definitive folds exist only within
development. The tracked manifest records configuration, data and assignment
hashes at `reports/validation/v2_grouped_folds_manifest.json`; the full local
assignment remains ignored at `artifacts/folds/v2_grouped_assignment.csv`.

Model selection used development OOF predictions only. The retained
`GB_RET20_REFERENCE` achieved accuracy 0.520005, ROC-AUC 0.526351, and log-loss
0.692253. Its exact configuration and report are tracked under `configs/` and
`reports/`. Advanced feature families and boosting engines remain historical
research; they are not the production pipeline.

The finalization procedure freezes the RET_1-to-RET_20 Gradient Boosting
configuration, evaluates it exactly once on the untouched lockbox, and then
fits a fresh pipeline on all labelled rows. No notebook is required to build,
evaluate, serve, or test the final model.
