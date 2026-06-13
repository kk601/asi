from contextlib import asynccontextmanager
import logging
from pathlib import Path
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
from typing import Literal
from autogluon.tabular import TabularPredictor

models = {}
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
AUTOML_MODEL_PATH = BASE_DIR / "data" / "06_models" / "autogluon"
BASELINE_MODEL_PATH = BASE_DIR / "data" / "06_models" / "baseline_model.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Funkcja uruchamiana przy starcie i zamykaniu aplikacji.
    Najpierw próbuje załadować model AutoGluon. Jeśli go nie ma lub wystąpi błąd,
    przechodzi do ładowania modelu baseline (fallback).
    """

    if AUTOML_MODEL_PATH.exists():
        try:
            models["satisfaction_model"] = TabularPredictor.load(str(AUTOML_MODEL_PATH))
            models["model_type"] = "AutoGluon"
            logger.info(f"Pomyślnie załadowano model AutoGluon z {AUTOML_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Nie udało się załadować modelu AutoGluon: {e}")

    if models.get("satisfaction_model") is None and BASELINE_MODEL_PATH.exists():
        try:
            with open(BASELINE_MODEL_PATH, "rb") as f:
                models["satisfaction_model"] = pickle.load(f)
            models["model_type"] = "Baseline"
            logger.info(f"Pomyślnie załadowano model baseline (fallback) z {BASELINE_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Nie udało się załadować modelu baseline: {e}")
    
    if models.get("satisfaction_model") is None:
        logger.warning("Nie załadowano żadnego modelu")

    yield
    
    # Czyszczenie zasobów przy wyłączaniu serwera
    models.clear()


app = FastAPI(
    title="Airline Passenger Satisfaction API",
    description="REST API do przewidywania satysfakcji pasażerów linii lotniczych.",
    version="1.1.0",
    lifespan=lifespan
)


class PassengerData(BaseModel):
    """
    Schemat danych wejściowych z walidacją typów i ograniczeniami (Pydantic).
    Używamy aliasów, aby obsłużyć spacje i myślniki w nazwach kolumn wejściowych.
    """
    # Zmienne kategoryczne
    Gender: Literal["Male", "Female"] = Field(..., description="Płeć (Male/Female)")
    Customer_Type: Literal["Loyal Customer", "disloyal Customer"] = Field(..., alias="Customer Type")
    Type_of_Travel: Literal["Personal Travel", "Business travel"] = Field(..., alias="Type of Travel")
    Class: Literal["Eco", "Eco Plus", "Business"] = Field(...)
    
    # Zmienne numeryczne
    Age: int = Field(..., ge=1, le=120, description="Wiek pasażera (1-120)")
    Flight_Distance: int = Field(..., ge=0, alias="Flight Distance")
    Departure_Delay: int = Field(0, ge=0, alias="Departure Delay")
    Arrival_Delay: float = Field(0.0, ge=0.0, alias="Arrival Delay")
    
    # Oceny ankietowe (skala 0-5)
    Departure_and_Arrival_Time_Convenience: int = Field(..., ge=0, le=5, alias="Departure and Arrival Time Convenience")
    Ease_of_Online_Booking: int = Field(..., ge=0, le=5, alias="Ease of Online Booking")
    Check_in_Service: int = Field(..., ge=0, le=5, alias="Check-in Service")
    Online_Boarding: int = Field(..., ge=0, le=5, alias="Online Boarding")
    Gate_Location: int = Field(..., ge=0, le=5, alias="Gate Location")
    On_board_Service: int = Field(..., ge=0, le=5, alias="On-board Service")
    Seat_Comfort: int = Field(..., ge=0, le=5, alias="Seat Comfort")
    Leg_Room_Service: int = Field(..., ge=0, le=5, alias="Leg Room Service")
    Cleanliness: int = Field(..., ge=0, le=5, alias="Cleanliness")
    Food_and_Drink: int = Field(..., ge=0, le=5, alias="Food and Drink")
    In_flight_Service: int = Field(..., ge=0, le=5, alias="In-flight Service")
    In_flight_Wifi_Service: int = Field(..., ge=0, le=5, alias="In-flight Wifi Service")
    In_flight_Entertainment: int = Field(..., ge=0, le=5, alias="In-flight Entertainment")
    Baggage_Handling: int = Field(..., ge=0, le=5, alias="Baggage Handling")

    # Pozwala Pydanticowi na akceptowanie nazw pythonowych i aliasów
    model_config = ConfigDict(populate_by_name=True)


@app.get("/health")
def health_check() -> dict:
    """
    Sprawdza stan API i status załadowanych modeli
    """
    model = models.get("satisfaction_model")
    model_type = models.get("model_type", "None")
    
    exact_model_name = "None"
    if model is not None:
        if model_type == "AutoGluon":
            exact_model_name = model.model_best
        else:
            if hasattr(model, "name"):
                exact_model_name = model.name
            else:
                exact_model_name = f"Baseline_{type(model).__name__}"
    
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": exact_model_name
    }


@app.post("/predict")
def predict(data: PassengerData) -> dict:
    """
    Endpoint wykonujący predykcję na podstawie przekazanych danych pasażera.
    """
    model = models.get("satisfaction_model")
    model_type = models.get("model_type")
    
    # 503 Service Unavailable, jeśli model nie jest załadowany
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded. Ensure the model file exists."
        )

    # Konwersja obiektu Pydantic do DataFrame. 
    input_df = pd.DataFrame([data.model_dump(by_alias=True)])

    # Standaryzacja nazw kolumn do tych z Kedro
    input_df.columns = input_df.columns.str.lower().str.replace(" ", "_")

    try:
        # Predykcja modelem
        prediction = model.predict(input_df)

        # Normalizacja autoGluon - sklearn
        pred_val = prediction.iloc[0] if hasattr(prediction, "iloc") else prediction[0]

        if model_type == "AutoGluon":
            exact_model_name = model.model_best
        else:
            if hasattr(model, "name"):
                exact_model_name = model.name
            else:
                exact_model_name = f"Baseline_{type(model).__name__}"

        return {
            "prediction": str(pred_val), 
            "model": exact_model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")