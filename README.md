# Housing Society Water Demand (Lucknow) - MP009

**Course**: SCSE3040 MLOps Mini Project  
**Group ID**: `G195`  
**Problem ID**: `MP009`  
**Faculty**: Saurabh Shrivastava | **Batch**: 32  

---

## 1. Group Members & Roles

| Enrolment No. | Name | Role | Core Ownership |
|---|---|---|---|
| **S24CSEU1450** | Nirnay Maiti | Service & Ops Lead | FastAPI prediction service, Pydantic validation rules, JSON logging, Docker, Kubernetes, CI/CD GitHub Actions, unit tests |
| **S24CSEU1444** | Sarvatha Shekhawat | Data & Drift Lead | DVC setup, 7 data quality checks, data cleaning logic, PSI feature drift calculation |
| **S24CSEU1445** | Mayank Gupta | Model & Registry Lead | MLflow tracking pipeline, candidate training, model registry, champion promotion, holdout predictions |

---

## 2. Project Overview

The **Lucknow Residents' Welfare Association** manages a large housing society in Lucknow that procures supplemental water via tankers when overhead reservoirs run low. Accurate daily water demand prediction (`litres_per_day`) enables the association to pre-order tankers 24 hours in advance at discounted rates, preventing supply disruption and emergency overhead charges.

This repository implements the end-to-end MLOps pipeline:
1. **Data Versioning (DVC)**: Tracking messy historical training data and post-deployment batches.
2. **Data Quality Auditing**: Systematic identification of data issues in `train.csv`.
3. **Reproducible Model Training & Tracking (MLflow)**: Multiple model configurations tracked to `mlflow.db`, evaluation using MAE, and model registry management.
4. **Drift Monitoring & Retraining**: Population Stability Index (PSI) calculation, performance degradation analysis per group (`tower`), retraining on arrival of `later.csv`, and automated champion promotion.
5. **Production Prediction API (FastAPI)**: REST API with validation rules and structured JSON logging.
6. **Containerization & Orchestration**: Production `Dockerfile` and Kubernetes `Deployment` & `Service` manifests.
7. **Continuous Integration**: GitHub Actions CI workflow executing pytest on every push.

---

## 3. Repository Structure

```
├── .dvc/                             # DVC configuration
├── .github/workflows/ci.yml          # GitHub Actions CI workflow (pytest)
├── app/
│   └── main.py                       # FastAPI application (GET /health, POST /predict)
├── data/
│   ├── raw/
│   │   ├── train.csv.dvc             # DVC metadata for original training data
│   │   ├── later.csv.dvc             # DVC metadata for monitoring data
│   │   └── holdout_features.csv      # Unlabeled holdout features
├── k8s/
│   ├── deployment.yaml               # Kubernetes deployment (2 replicas, health probes)
│   └── service.yaml                  # Kubernetes ClusterIP service (port 80 -> 8000)
├── models/
│   └── model.joblib                  # Serialized champion model (@champion)
├── predictions/
│   └── holdout_predictions.csv       # Final predictions on holdout features
├── reports/
│   └── findings.yaml                 # Form filled with audit and experiment metrics
├── src/
│   └── water_demand/
│       ├── __init__.py
│       ├── clean.py                  # 7 data issue checks & data cleaning routines
│       ├── monitoring.py             # PSI computation & drift analysis
│       └── train.py                  # End-to-end MLflow training & evaluation pipeline
├── tests/
│   ├── test_api.py                   # FastAPI TestClient endpoint & rule tests
│   └── test_clean.py                 # Pipeline data cleaning & PSI tests
├── Dockerfile                        # Container recipe (python:3.12-slim)
├── .dockerignore                     # Docker build exclusions
├── findings.yaml                     # Root findings form
├── mlflow.db                         # SQLite tracking database
├── params.yaml                       # Pipeline configurations and hyperparameters
├── requirements.txt                  # Pinned dependencies
└── README.md                         # Project documentation
```

---

## 4. Data Quality Audits & Cleaning

An audit of the seven potential data problems in `train.csv` revealed:

1. **Unit Mixup (`present: true`)**: `supply_hours` for **Tower B** was recorded in minutes (range 60.0 – 714.0) instead of hours (range 1.0 – 12.0). Corrected by dividing Tower B values by `60`.
2. **Duplicate Rows (`present: true`)**: Detected **38** redundant duplicate rows. Corrected using deduplication (`drop_duplicates()`), reducing rows from 1,038 to 1,000.
3. **Target Typos (`present: true`)**: Detected **13** records with values > 100,000 litres/day caused by typing an extra trailing zero (e.g. 171,620 instead of 17,162). Corrected by dividing by `10`.
4. **Leaking Column (`present: false`)**: All training features match `holdout_features.csv`.
5. **Sentinel Value (`present: false`)**: No sentinel codes (-1, 999, etc.) detected.
6. **Stuck Sensor (`present: false`)**: No multi-week sensor freeze runs found.
7. **Spelling Variants (`present: false`)**: Categorical labels in `supply_source` and `tower` match canonical levels.

---

## 5. Drift Monitoring & Retraining (v1 vs v2)

- **Population Stability Index (PSI)**: Evaluated across 10 decile bins between cleaned `train.csv` and `later.csv`.
  - `temp_c`: **2.8778** (Extreme Drift due to seasonal monsoon shift).
  - All other features remained stable (PSI < 0.08).
  - **Most Drifted Feature**: `temp_c`.
- **Group Disparity**: v1 model evaluation on `later.csv` revealed **Tower B** was the worst-served group with an MAE of **18,173.30 litres** (vs ~2,500 litres for Towers A and C).
- **Retraining (v2)**: Retrained on cleaned `train.csv` + first 70% of `later.csv`.
  - On the held-out 30% check set:
    - v1 MAE: **8,245.16**
    - v2 MAE: **4,902.36**
    - **Relative Error Reduction**: **40.54%**
  - **Decision**: `promote` (exceeds the 10% threshold rule). Model v2 was registered and assigned the `@champion` alias.

---

## 6. How to Run Locally

### Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Pipeline Execution
```bash
# Execute training, tracking, and holdout predictions
$env:PYTHONPATH="src;."
python -m water_demand.train
```

### Run Tests
```bash
pytest -v tests/
```

### Run API Service
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker Build & Run
```bash
docker build -t mp009-api:1.0 .
docker run -p 8000:8000 mp009-api:1.0
```
