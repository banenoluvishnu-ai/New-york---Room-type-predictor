# from fastapi import FastAPI
# import pandas as pd
# from pydantic import BaseModel, Field
# import joblib
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
# # from pathlib import Path
# # from fastapi.responses import FileResponse
# # from fastapi.staticfiles import StaticFiles


# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# COLUMNS = ["latitude", "longitude", "price", "minimum_nights",
#     "number_of_reviews", "reviews_per_month",
#     "calculated_host_listings_count", "availability_365",
#     "neighbourhood_group", "neighbourhood",]

# model = joblib.load("Model_Pipeline.pkl")  # Load the pre-trained model pipeline

# #Pydantic Model = the input validation
# class Features(BaseModel):
#     latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
#     longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
#     price: float = Field(..., gt=0, description="Price per night, must be positive")
#     minimum_nights: int = Field(..., ge=1, le=365, description="Minimum nights required for booking")
#     number_of_reviews: int = Field(..., ge=0, description="Total number of reviews")
#     reviews_per_month: float = Field(..., ge=0, description="Average reviews per month")
#     calculated_host_listings_count: int = Field(..., ge=0, description="Number of listings by this host")
#     availability_365: int = Field(..., ge=0, le=365, description="Days available out of 365")
#     neighbourhood_group: str = Field(..., min_length=1, description="Borough or neighbourhood group")
#     neighbourhood: str = Field(..., min_length=1, description="Specific neighbourhood name")



# # @app.get('/')
# # def greet():
# #     return "Hello Guyss"

# # from pathlib import Path
# # from fastapi.responses import FileResponse

# # BASE_DIR = Path(__file__).resolve().parent

# # @app.get("/")
# # def home():
# #     return FileResponse(BASE_DIR / "index.html")


# from pathlib import Path
# from fastapi.responses import FileResponse

# BASE_DIR = Path(__file__).resolve().parent

# @app.get("/")
# def greet():
#     return FileResponse(BASE_DIR / "index.html")


# @app.post('/predict')
# def predict(features: Features):
#     row = pd.DataFrame([features.dict()], columns=COLUMNS)
#     prediction  = model.predict(row)
#     probability = model.predict_proba(row)

#     return {
#         "Predicted_room_type": prediction[0],
#         "Probability": probability.tolist()[0]}




from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

COLUMNS = [
    "latitude", "longitude", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "neighbourhood_group", "neighbourhood",
]

# Load the pre-trained model pipeline
model = joblib.load(BASE_DIR / "Model_Pipeline.pkl")


# Pydantic model = input validation
class Features(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    price: float = Field(..., gt=0, description="Price per night, must be positive")
    minimum_nights: int = Field(..., ge=1, le=365, description="Minimum nights required for booking")
    number_of_reviews: int = Field(..., ge=0, description="Total number of reviews")
    reviews_per_month: float = Field(..., ge=0, description="Average reviews per month")
    calculated_host_listings_count: int = Field(..., ge=0, description="Number of listings by this host")
    availability_365: int = Field(..., ge=0, le=365, description="Days available out of 365")
    neighbourhood_group: str = Field(..., min_length=1, description="Borough or neighbourhood group")
    neighbourhood: str = Field(..., min_length=1, description="Specific neighbourhood name")


# ---------- Frontend ----------
@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/style.css")
def css():
    return FileResponse(BASE_DIR / "style.css", media_type="text/css")


@app.get("/script.js")
def js():
    return FileResponse(BASE_DIR / "script.js", media_type="application/javascript")


# ---------- API ----------
@app.post("/predict")
def predict(features: Features):
    row = pd.DataFrame([features.model_dump()], columns=COLUMNS)
    prediction = model.predict(row)
    probability = model.predict_proba(row)

    return {
        "Predicted_room_type": str(prediction[0]),
        "Probability": probability.tolist()[0],
    }

