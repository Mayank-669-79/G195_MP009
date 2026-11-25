
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def calculate_psi(expected: pd.Series, actual: pd.Series, num_bins: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between expected (train) and actual (later).
    Uses 10 quantile bins based on the expected distribution.
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, num_bins + 1))
    breakpoints = np.unique(breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    exp_counts = pd.cut(expected, bins=list(breakpoints)).value_counts(sort=False)
    act_counts = pd.cut(actual, bins=list(breakpoints)).value_counts(sort=False)

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Avoid zero division and log(0)
    exp_pct = exp_pct.replace(0, 1e-4)
    act_pct = act_pct.replace(0, 1e-4)

    psi_val = ((act_pct - exp_pct) * np.log(act_pct / exp_pct)).sum()
    return float(round(psi_val, 4))


def compute_feature_drift(
    train_df: pd.DataFrame,
    later_df: pd.DataFrame,
    numeric_features: list,
    num_bins: int = 10
) -> tuple[dict[str, float], str]:
    """
    Computes PSI for each numeric feature and identifies the most drifted feature.
    """
    psi_dict = {}
    for col in numeric_features:
        psi_dict[col] = calculate_psi(train_df[col], later_df[col], num_bins=num_bins)

    most_drifted = max(psi_dict, key=lambda k: psi_dict[k])
    return psi_dict, most_drifted


def compute_mae_by_group(
    y_true: pd.Series,
    y_pred: Any,
    groups: pd.Series
) -> tuple[dict[str, float], str]:
    """
    Computes MAE broken down by group (e.g. tower) and identifies the worst-served group.
    """
    mae_dict = {}
    for group_val in sorted(groups.unique()):
        idx = (groups == group_val)
        mae_val = mean_absolute_error(y_true[idx], y_pred[idx])
        mae_dict[str(group_val)] = float(round(mae_val, 2))

    worst_group = max(mae_dict, key=lambda k: mae_dict[k])
    return mae_dict, worst_group


def evaluate_promotion(
    v1_mae: float,
    v2_mae: float,
    threshold_pct: float = 0.10
) -> tuple[str, float]:
    """
    Decides whether to promote v2:
    Promote v2 only if its MAE on check set is at least 10% lower than v1's.
    Returns decision ('promote' or 'keep') and relative reduction.
    """
    reduction = (v1_mae - v2_mae) / v1_mae
    if reduction >= threshold_pct:
        decision = "promote"
    else:
        decision = "keep"
    return decision, float(round(reduction, 4))
