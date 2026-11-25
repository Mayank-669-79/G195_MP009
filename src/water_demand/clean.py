from typing import Any

import pandas as pd


def detect_data_issues(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    holdout_df: pd.DataFrame
) -> dict[str, Any]:
    """
    Audits train.csv for the seven kinds of data problems specified in STUDENT_HANDBOOK.md.
    Returns a dictionary matching the findings.yaml data_issues structure.
    """
    issues: dict[str, Any] = {}

    # 1. Leaking column
    train_feature_cols = [c for c in train_df.columns if c not in ['record_id', 'date', 'litres_per_day']]
    missing_in_holdout = [c for c in train_feature_cols if c not in holdout_df.columns]
    if missing_in_holdout:
        issues['leaking_column'] = {
            'present': True,
            'column': missing_in_holdout[0]
        }
    else:
        issues['leaking_column'] = {
            'present': False,
            'column': None
        }

    # 2. Unit mixup
    # Compare minimum, average, and maximum for each group column (tower)
    unit_mixup_found = False
    numeric_cols = train_df.select_dtypes(include='number').columns.tolist()
    if 'litres_per_day' in numeric_cols:
        numeric_cols.remove('litres_per_day')

    for col in numeric_cols:
        group_means = train_df.groupby('tower')[col].mean()
        min_mean = group_means.min()
        max_mean = group_means.max()
        ratio = max_mean / (min_mean + 1e-9)
        # Check standard conversion factors: 10, 12, 60, 100, 1000
        for factor in [10, 12, 60, 100, 1000]:
            if 0.8 * factor <= ratio <= 1.2 * factor:
                affected_group = group_means.idxmax()
                issues['unit_mixup'] = {
                    'present': True,
                    'column': col,
                    'group_value': affected_group,
                    'factor': factor
                }
                unit_mixup_found = True
                break
        if unit_mixup_found:
            break

    if not unit_mixup_found:
        issues['unit_mixup'] = {
            'present': False,
            'column': None,
            'group_value': None,
            'factor': None
        }

    # 3. Sentinel value
    sentinel_found = False
    for col in numeric_cols:
        vc = train_df[col].value_counts()
        for sentinel in [-1, -99, -999, 999, 9999]:
            if sentinel in vc.index and vc[sentinel] > 10:
                issues['sentinel_value'] = {
                    'present': True,
                    'column': col,
                    'value': int(sentinel),
                    'count': int(vc[sentinel])
                }
                sentinel_found = True
                break
        if sentinel_found:
            break

    if not sentinel_found:
        issues['sentinel_value'] = {
            'present': False,
            'column': None,
            'value': None,
            'count': None
        }

    # 4. Duplicate rows
    dup_count = int(train_df.duplicated().sum())
    issues['duplicate_rows'] = {
        'present': dup_count > 0,
        'count': dup_count if dup_count > 0 else None
    }

    # 5. Stuck sensor
    stuck_found = False
    train_sorted = train_df.sort_values('date')
    for col in numeric_cols:
        s = train_sorted[col]
        run_id = (s != s.shift()).cumsum()
        runs = train_sorted.groupby(run_id).agg(
            val=(col, 'first'),
            count=(col, 'count'),
            start_date=('date', 'first'),
            end_date=('date', 'last')
        )
        long_runs = runs[runs['count'] >= 14]  # 2+ weeks of consecutive identical readings
        if len(long_runs) > 0:
            row = long_runs.iloc[0]
            issues['stuck_sensor'] = {
                'present': True,
                'column': col,
                'start_date': str(row['start_date']),
                'end_date': str(row['end_date'])
            }
            stuck_found = True
            break

    if not stuck_found:
        issues['stuck_sensor'] = {
            'present': False,
            'column': None,
            'start_date': None,
            'end_date': None
        }

    # 6. Target typos
    # Find huge jump (e.g. > 100,000 when normal max is ~60,000)
    typo_mask = train_df['litres_per_day'] > 100000
    typo_count = int(typo_mask.sum())
    issues['target_typos'] = {
        'present': typo_count > 0,
        'count': typo_count if typo_count > 0 else None
    }

    # 7. Spelling variants
    spelling_found = False
    text_cols = train_df.select_dtypes(include='object').columns.tolist()
    for col in text_cols:
        if col in ['record_id', 'date']:
            continue
        train_cats = set(train_df[col].dropna().unique())
        later_cats = set(later_df[col].dropna().unique())
        # Check if train has variants not in clean later
        if train_cats != later_cats and len(train_cats) > len(later_cats):
            issues['spelling_variants'] = {
                'present': True,
                'column': col,
                'true_levels': len(later_cats)
            }
            spelling_found = True
            break

    if not spelling_found:
        issues['spelling_variants'] = {
            'present': False,
            'column': None,
            'true_levels': None
        }

    return issues


def clean_train_data(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw training dataframe by:
    1. Deduplicating rows
    2. Correcting unit mixup in Tower B supply_hours (minutes -> hours)
    3. Correcting target typos (dividing by 10)
    """
    df = train_df.drop_duplicates().copy()

    # Fix unit mixup in Tower B supply_hours
    mask_b = df['tower'] == 'Tower B'
    # In Tower B, values were entered in minutes (~60 - 714) rather than hours (~1 - 12)
    df.loc[mask_b, 'supply_hours'] = df.loc[mask_b, 'supply_hours'] / 60.0

    # Fix target typos (values > 100,000 had an extra zero typed)
    mask_typo = df['litres_per_day'] > 100000
    df.loc[mask_typo, 'litres_per_day'] = (df.loc[mask_typo, 'litres_per_day'] / 10.0).round().astype(int)

    return df
