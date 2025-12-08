# Data Preprocessing Guide

## What the script does (`data-prerprocessing.py`)
- Loads `mental-health-data/train.csv` and `mental-health-data/test.csv`.
- Splits features/labels, drops `id`/`Name`, keeps `Depression` as `y`.
- Cleans missing values: job/study columns and all numerics -> `0`; categoricals -> `Unknown`.
- Normalizes key columns:
  - `Degree` → ordered buckets (`HighSchool` < `Bachelor` < `Master` < `PhD` < `Other`).
  - `Sleep Duration` → numeric hours (parses ranges/phrases; fills missing with global median).
  - `Dietary Habits` → ordered health levels.
  - `Gender` → binary (`male`=1, `female`=0, other/unknown=-1).
- Encodes:
  - Ordinal for cleaned degree/diet.
  - Binary: suicidal thoughts, family history, working/student, gender.
  - One-hot: `City`, `Profession`.
- Keeps binary columns unscaled; scales remaining numerics with `MinMaxScaler`.
- Removes outliers from `Age`, `Work/Study Hours`, `Sleep_Hours` via IQR filter.
- Train/valid split (80/20, `random_state=42`) after outlier drop.
- Saves processed splits and targets to CSV.

## Outputs (already generated)
- `X_train_processed.csv`, `X_valid_processed.csv`, `X_test_processed.csv`
- `y_train.csv`, `y_valid.csv`
- `test_id.csv`

## How to reproduce
1) Install deps: `pip install -r requirement.txt`
2) Ensure raw data present: `mental-health-data/train.csv`, `mental-health-data/test.csv`
3) Run: `python data-prerprocessing.py`
4) Check outputs in repo root as listed above.

## Notes for collaborators
- If you add new categorical fields, append them to `categorical_cols` before the transformer.
- For new binary/yes-no fields, map them like the existing `binary_cols` and exclude from scaling.
- Update the outlier column list (`cols_for_outliers`) if you want filtering on additional continuous fields.
- Be mindful of the ordinal order arrays (`deg_order`, `diet_order`) if labels change.
