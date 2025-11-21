
import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------- Paths ----------
DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

PLACES_PATH = os.path.join(DATA_DIR, "Places.csv")
CITY_PATH = os.path.join(DATA_DIR, "City.csv")

PLACES_DF_PATH = os.path.join(MODELS_DIR, "places_df.pkl")
CITY_DF_PATH = os.path.join(MODELS_DIR, "city_df.pkl")
SENTENCE_MODEL_DIR = os.path.join(MODELS_DIR, "sentence_model")

# ---------- Load ----------
places_df = pd.read_csv(PLACES_PATH)
city_df = pd.read_csv(CITY_PATH)

# ---------- Cleaning ----------
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\(\[].*?[\)\]]", " ", s)
    return s.strip()

def fix_brackets(text):
    if isinstance(text, str):
        text = text.strip()
        if text.startswith("[") and text.endswith("]"):
            return text[1:-1].strip()
    return text

places_df["Place"] = (
    places_df["Place"].astype(str)
    .str.replace(r"^\s*\d+\.\s*", "", regex=True)
    .str.strip()
)

places_df = places_df.rename(columns={"Ratings": "Place_Rating", "Place_desc": "Place_Desc"})
city_df = city_df.rename(columns={"Ratings": "City_Rating", "City_desc": "City_Desc"})

places_df["Place_Desc"] = places_df["Place_Desc"].fillna("").map(clean_text)
city_df["City_Desc"] = city_df["City_Desc"].fillna("").map(fix_brackets).map(clean_text)

places_df["Place_Rating"] = places_df["Place_Rating"].fillna(0)
city_df["City_Rating"] = city_df["City_Rating"].fillna(0)

# ---------- SentenceTransformer ----------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding cities...")
# converts the column of city descriptions into a Python list and generate embeddings for each description
city_df["embedding"] = list(model.encode(city_df["City_Desc"].tolist(), show_progress_bar=True))

print("Encoding places...")
places_df["embedding"] = list(model.encode(places_df["Place_Desc"].tolist(), show_progress_bar=True))

# ---------- Save cleaned + embedded data ----------
places_df.to_pickle(PLACES_DF_PATH)
city_df.to_pickle(CITY_DF_PATH)

# ---------- Save model folder for Render ----------
print("Saving model...")
model.save(SENTENCE_MODEL_DIR)


