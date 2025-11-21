from fastapi import FastAPI
from pydantic import BaseModel # FastAPI uses Pydantic to make sure everything is clean and structured
from typing import List, Optional
import pandas as pd
import os

from utils.recommender import HybridRecommender

# ------------------ PATHS ------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

PLACES_DF_PATH = os.path.join(MODELS_DIR, "places_df.pkl")
CITY_DF_PATH = os.path.join(MODELS_DIR, "city_df.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "sentence_model")


# ------------------ LOAD ARTIFACTS -> data containing travel info ------------------

places_df = pd.read_pickle(PLACES_DF_PATH)
city_df = pd.read_pickle(CITY_DF_PATH)

# Create recommender model
recommender = HybridRecommender(
    places_df=places_df,
    city_df=city_df,
    model_path=MODEL_PATH
)


# ------------------ FASTAPI APP ------------------

app = FastAPI(
    title="Travel Recommender API",
    description="Hybrid semantic recommender for finding top places based on user travel intent.",
    version="1.0.0",
)


# ------------------ REQUEST + RESPONSE MODELS ------------------

class RecommendRequest(BaseModel):
    query: str
    n_cities: int = 3           # What the user can send
    n_places_each: int = 5
    show_desc: bool = True


class Recommendation(BaseModel):
    City: str                   # What the API will send back
    Place: str
    Rating: float
    Similarity: float
    Description: Optional[str] = None


class RecommendResponse(BaseModel):
    query: str
    results: List[Recommendation]


# ----------------- CREATE API ROUTES ------------------

@app.get("/")  # welcome msg
def home():
    return {
        "message": "Travel Recommender API is running!",
        "docs": "/docs"
    }

# takes input and return best travel suggestion
@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(request: RecommendRequest):

    results = recommender.recommend(
        query=request.query,
        n_cities=request.n_cities,
        n_places_each=request.n_places_each,
        show_desc=request.show_desc,
    )

    return RecommendResponse(
        query=request.query,
        results=results
    )


###  how to execute this fastapi
# In terminal --
# uvicorn main:app --reload   -- uvicorn only starts the server doesnot display the webapages
# so run this on the browser to open URL
# http://localhost:8000/docs   --- FastAPI's Swagger UI 

# or just do /docs after clicking on the link Uvicorn running on http://127.0.0.1:8000


# try this in the query 
#  "query": "hillside adventure trekking" , "relaxing beach vacation, water sports, tropical vibes"
