import hashlib
import os
import sys
from pathlib import Path

# Ensure src/ and repo root are always in sys.path
_src_dir = Path(__file__).resolve().parent.parent
_root_dir = _src_dir.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from water_demand.clean import clean_train_data, detect_data_issues
from water_demand.monitoring import (
    compute_feature_drift,
    compute_mae_by_group,
    evaluate_promotion,
)


def compute_sha256(filepath: str) -> str:
    """Computes the SHA256 checksum of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def build_pipeline(model):
    """Builds a scikit-learn pipeline with preprocessor and regressor."""
    num_features = ['flats_occupied', 'temp_c', 'garden_area_m2', 'tank_level_morning_pct', 'leak_complaints', 'supply_hours']
    cat_features = ['supply_source', 'tower']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_features)
        ]
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])


def main():
    # Load configuration
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    raw_train_path = params['data']['raw_train_path']
    raw_later_path = params['data']['raw_later_path']
    holdout_path = params['data']['holdout_features_path']
    tracking_uri = params['tracking']['tracking_uri']
    experiment_name = params['tracking']['experiment_name']
    reg_model_name = params['tracking']['registered_model_name']
    features = params['model']['features']
    target = params['model']['target_col']
    num_features = params['model']['numeric_features']

    # 1. Compute original raw train SHA-256
    train_sha256 = compute_sha256(raw_train_path)
    print(f"Original train.csv SHA-256: {train_sha256}")

    # Load raw datasets
    train_raw = pd.read_csv(raw_train_path)
    later_raw = pd.read_csv(raw_later_path)
    holdout_raw = pd.read_csv(holdout_path)

    # 2. Audit and detect 7 data issues
    data_issues = detect_data_issues(train_raw, later_raw, holdout_raw)
    print("\nAudited Data Issues:")
    for k, v in data_issues.items():
        print(f"  {k}: {v}")

    # 3. Clean training data
    train_clean = clean_train_data(train_raw)
    print(f"\nCleaned train rows: {len(train_clean)} (raw: {len(train_raw)})")

    # Set up MLflow
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    X_train = train_clean[features]
    y_train = train_clean[target]
    X_later = later_raw[features]
    y_later = later_raw[target]

    # Candidate settings for v1
    candidate_models = {
        "ridge_regressor": (Ridge(alpha=1.0), {"alpha": 1.0}),
        "random_forest": (RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42), {"n_estimators": 100, "max_depth": 10}),
        "gradient_boosting": (GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42), {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1})
    }

    best_v1_run_id = None
    best_v1_mae = float("inf")
    best_v1_pipe = None

    print("\n=== Training v1 Candidates ===")
    for model_name, (model_obj, hyperparams) in candidate_models.items():
        pipe = build_pipeline(model_obj)
        with mlflow.start_run(run_name=f"v1_{model_name}") as run:
            run_id = run.info.run_id
            mlflow.log_param("data_sha256", train_sha256)
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(hyperparams)

            pipe.fit(X_train, y_train)

            train_mae = mean_absolute_error(y_train, pipe.predict(X_train))
            later_mae = mean_absolute_error(y_later, pipe.predict(X_later))

            mlflow.log_metric("mae_train", float(round(train_mae, 2)))
            mlflow.log_metric("mae_later", float(round(later_mae, 2)))

            mlflow.sklearn.log_model(pipe, "model", serialization_format="cloudpickle")
            print(f"Run {model_name} (ID: {run_id}) -> Train MAE: {train_mae:.2f}, Later MAE: {later_mae:.2f}")

            if later_mae < best_v1_mae:
                best_v1_mae = later_mae
                best_v1_run_id = run_id
                best_v1_pipe = pipe

    print(f"\nBest v1 Model: Run ID {best_v1_run_id} with Later MAE {best_v1_mae:.2f}")

    # Register best v1 model as Version 1
    model_uri = f"runs:/{best_v1_run_id}/model"
    v1_mv = mlflow.register_model(model_uri, reg_model_name)
    v1_version = int(v1_mv.version)
    print(f"Registered {reg_model_name} Version {v1_version}")

    # 4. Monitoring: Compute PSI between cleaned train.csv and later.csv
    psi_dict, most_drifted_feature = compute_feature_drift(train_clean, later_raw, num_features, num_bins=10)
    print("\nFeature PSI values (train vs later):")
    for feat, val in psi_dict.items():
        print(f"  {feat}: {val}")
    print(f"Most drifted feature: {most_drifted_feature}")

    # 5. Compute v1's MAE on later.csv separately for each tower
    assert best_v1_pipe is not None
    v1_later_preds = best_v1_pipe.predict(X_later)
    mae_by_group, worst_served_group = compute_mae_by_group(y_later, v1_later_preds, later_raw['tower'])
    print("\nv1 MAE by tower on later.csv:")
    for grp, val in mae_by_group.items():
        print(f"  {grp}: {val:.2f}")
    print(f"Worst-served group: {worst_served_group}")

    # 6. Retraining v2:
    # Sort later.csv by date. Split first 70% and last 30%.
    later_sorted = later_raw.sort_values('date').reset_index(drop=True)
    split_idx = int(len(later_sorted) * 0.7)
    later_70 = later_sorted.iloc[:split_idx].copy()
    later_30 = later_sorted.iloc[split_idx:].copy()

    train_v2 = pd.concat([train_clean, later_70], ignore_index=True)
    X_train_v2 = train_v2[features]
    y_train_v2 = train_v2[target]

    X_check = later_30[features]
    y_check = later_30[target]

    # Retrain v2
    v2_pipe = build_pipeline(
        GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    )

    with mlflow.start_run(run_name="v2_retrained") as run:
        v2_run_id = run.info.run_id
        mlflow.log_param("data_sha256", train_sha256)
        mlflow.log_param("model_name", "v2_retrained_gradient_boosting")
        mlflow.log_param("retrain_split", 0.70)

        v2_pipe.fit(X_train_v2, y_train_v2)

        v2_train_mae = mean_absolute_error(y_train_v2, v2_pipe.predict(X_train_v2))
        mlflow.log_metric("mae_train", float(round(v2_train_mae, 2)))

        # Compare v1 and v2 on check set (last 30%)
        v1_check_preds = best_v1_pipe.predict(X_check)
        v2_check_preds = v2_pipe.predict(X_check)

        mae_v1_on_check = float(round(mean_absolute_error(y_check, v1_check_preds), 2))
        mae_v2_on_check = float(round(mean_absolute_error(y_check, v2_check_preds), 2))

        mlflow.log_metric("mae_v1_check", mae_v1_on_check)
        mlflow.log_metric("mae_v2_check", mae_v2_on_check)

        mlflow.sklearn.log_model(v2_pipe, "model", serialization_format="cloudpickle")

    # Register v2
    v2_mv = mlflow.register_model(f"runs:/{v2_run_id}/model", reg_model_name)
    v2_version = int(v2_mv.version)
    print(f"Registered {reg_model_name} Version {v2_version}")

    # Evaluate promotion rule
    decision, reduction_pct = evaluate_promotion(mae_v1_on_check, mae_v2_on_check, threshold_pct=0.10)
    print("\nPromotion Evaluation on check set:")
    print(f"  v1 MAE on check: {mae_v1_on_check}")
    print(f"  v2 MAE on check: {mae_v2_on_check}")
    print(f"  Relative reduction: {reduction_pct * 100:.2f}%")
    print(f"  Decision: {decision}")

    if decision == "promote":
        champion_version = v2_version
        champion_model = v2_pipe
        client.set_registered_model_alias(reg_model_name, "champion", str(v2_version))
        print(f"Set alias '@champion' on Version {v2_version}")
    else:
        champion_version = v1_version
        champion_model = best_v1_pipe
        client.set_registered_model_alias(reg_model_name, "champion", str(v1_version))
        print(f"Set alias '@champion' on Version {v1_version}")

    # Save champion model for production serving
    assert champion_model is not None
    os.makedirs("models", exist_ok=True)
    joblib.dump(champion_model, params['model']['champion_model_path'])
    print(f"Saved champion model to {params['model']['champion_model_path']}")

    # 7. Predict holdout set using champion model
    os.makedirs("predictions", exist_ok=True)
    holdout_features = holdout_raw[features]
    holdout_preds = champion_model.predict(holdout_features)

    holdout_pred_df = pd.DataFrame({
        "record_id": holdout_raw["record_id"],
        "prediction": np.round(holdout_preds, 2)
    })
    holdout_pred_df.to_csv(params['data']['predictions_path'], index=False)
    print(f"Saved holdout predictions to {params['data']['predictions_path']} ({len(holdout_pred_df)} rows)")

    # 8. Save findings.yaml
    findings_content = {
        "group_id": "G195",
        "problem_id": "MP009",
        "data_issues": data_issues,
        "tracking": {
            "experiment_name": experiment_name,
            "best_run_id": best_v1_run_id,
            "registered_model_name": reg_model_name,
            "champion_version": champion_version
        },
        "monitoring": {
            "psi": psi_dict,
            "most_drifted_feature": most_drifted_feature,
            "mae_by_group": mae_by_group,
            "worst_served_group": worst_served_group,
            "mae_v1_on_check": mae_v1_on_check,
            "mae_v2_on_check": mae_v2_on_check,
            "decision": decision
        }
    }

    # Write findings.yaml at root and reports/
    with open("findings.yaml", "w") as f:
        yaml.dump(findings_content, f, sort_keys=False)
    os.makedirs("reports", exist_ok=True)
    with open("reports/findings.yaml", "w") as f:
        yaml.dump(findings_content, f, sort_keys=False)
    print("Updated findings.yaml and reports/findings.yaml successfully.")


if __name__ == "__main__":
    main()
