import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# 1. Structured JSON Logging Setup (P10)
class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_record.update(record.extra_data)
        return json.dumps(log_record)


logger = logging.getLogger("water_demand_api")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(JsonFormatter())
logger.handlers = [stream_handler]


# 2. Model Global Reference & Lifespan
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load("models/model.joblib")
        logger.info("Loaded champion model from models/model.joblib")
    except (FileNotFoundError, OSError, ValueError) as e:
        logger.warning(f"Could not load champion model: {e}. Model will need to be present before /predict is called.")
    yield


app = FastAPI(title="Water Demand Prediction API", lifespan=lifespan)


# 3. Pydantic Request & Response Schemas
class WaterDemandRequest(BaseModel):
    flats_occupied: int = Field(..., description="Flats with people living")
    temp_c: float = Field(..., le=70.0, description="Maximum temperature; values above 70 are rejected")
    garden_area_m2: float = Field(..., ge=0.0, description="Garden area watered; negative values are rejected")
    tank_level_morning_pct: float = Field(..., description="Overhead tank level at 6am")
    leak_complaints: int = Field(..., description="Open leak complaints")
    supply_hours: float = Field(..., description="Hours of municipal supply")
    supply_source: Literal["municipal", "borewell", "tanker"] = Field(..., description="Main water source")
    tower: Literal["Tower A", "Tower B", "Tower C"] = Field(..., description="Tower name")


class WaterDemandResponse(BaseModel):
    prediction: float


# 4. Endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=WaterDemandResponse)
def predict(request: WaterDemandRequest):
    global model
    if model is None:
        try:
            model = joblib.load("models/model.joblib")
        except (FileNotFoundError, OSError, ValueError) as e:
            raise HTTPException(status_code=500, detail=f"Model not loaded: {e!s}")

    input_data = pd.DataFrame([{
        "flats_occupied": request.flats_occupied,
        "temp_c": request.temp_c,
        "garden_area_m2": request.garden_area_m2,
        "tank_level_morning_pct": request.tank_level_morning_pct,
        "leak_complaints": request.leak_complaints,
        "supply_hours": request.supply_hours,
        "supply_source": request.supply_source,
        "tower": request.tower
    }])

    pred = float(model.predict(input_data)[0])

    # Structured JSON log per prediction
    extra_log = {
        "event": "prediction",
        "tower": request.tower,
        "prediction_litres": round(pred, 2)
    }
    logger.info("Prediction generated", extra={"extra_data": extra_log})

    return WaterDemandResponse(prediction=round(pred, 2))
