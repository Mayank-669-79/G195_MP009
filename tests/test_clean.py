import numpy as np
import pandas as pd

from water_demand.clean import clean_train_data
from water_demand.monitoring import calculate_psi, evaluate_promotion


def test_clean_removes_duplicates():
    df = pd.DataFrame({
        "record_id": ["R1", "R1", "R2"],
        "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
        "flats_occupied": [100, 100, 120],
        "temp_c": [20.0, 20.0, 22.0],
        "garden_area_m2": [500, 500, 600],
        "tank_level_morning_pct": [50, 50, 60],
        "leak_complaints": [2, 2, 1],
        "supply_hours": [5.0, 5.0, 6.0],
        "supply_source": ["municipal", "municipal", "tanker"],
        "tower": ["Tower A", "Tower A", "Tower B"],
        "litres_per_day": [25000, 25000, 30000]
    })
    cleaned = clean_train_data(df)
    assert len(cleaned) == 2
    assert "R1" in cleaned["record_id"].values
    assert "R2" in cleaned["record_id"].values


def test_clean_fixes_tower_b_supply_hours():
    df = pd.DataFrame({
        "record_id": ["R1", "R2"],
        "date": ["2026-01-01", "2026-01-02"],
        "flats_occupied": [100, 120],
        "temp_c": [20.0, 22.0],
        "garden_area_m2": [500, 600],
        "tank_level_morning_pct": [50, 60],
        "leak_complaints": [2, 1],
        "supply_hours": [6.0, 360.0],  # Tower B has 360 minutes
        "supply_source": ["municipal", "tanker"],
        "tower": ["Tower A", "Tower B"],
        "litres_per_day": [25000, 30000]
    })
    cleaned = clean_train_data(df)
    # Tower A should remain 6.0
    assert cleaned.loc[cleaned["tower"] == "Tower A", "supply_hours"].iloc[0] == 6.0
    # Tower B should be converted: 360 / 60 = 6.0
    assert cleaned.loc[cleaned["tower"] == "Tower B", "supply_hours"].iloc[0] == 6.0


def test_clean_fixes_target_typos():
    df = pd.DataFrame({
        "record_id": ["R1", "R2"],
        "date": ["2026-01-01", "2026-01-02"],
        "flats_occupied": [100, 120],
        "temp_c": [20.0, 22.0],
        "garden_area_m2": [500, 600],
        "tank_level_morning_pct": [50, 60],
        "leak_complaints": [2, 1],
        "supply_hours": [6.0, 6.0],
        "supply_source": ["municipal", "tanker"],
        "tower": ["Tower A", "Tower B"],
        "litres_per_day": [25000, 350000]  # Extra zero typo: 350000
    })
    cleaned = clean_train_data(df)
    assert cleaned.loc[cleaned["record_id"] == "R2", "litres_per_day"].iloc[0] == 35000


def test_psi_identical_distributions():
    s1 = pd.Series(np.linspace(10, 100, 500))
    s2 = pd.Series(np.linspace(10, 100, 500))
    psi = calculate_psi(s1, s2, num_bins=10)
    assert psi < 0.01


def test_evaluate_promotion_rule():
    # If reduction >= 10% -> promote
    decision, red = evaluate_promotion(v1_mae=100.0, v2_mae=85.0, threshold_pct=0.10)
    assert decision == "promote"
    assert red == 0.15

    # If reduction < 10% -> keep
    decision, red = evaluate_promotion(v1_mae=100.0, v2_mae=95.0, threshold_pct=0.10)
    assert decision == "keep"
    assert red == 0.05
