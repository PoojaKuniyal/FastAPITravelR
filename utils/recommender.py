import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class HybridRecommender:

    def __init__(self, places_df, city_df, model_path):
        self.places_df = places_df
        self.city_df = city_df

        # Load saved ST model
        self.model = SentenceTransformer(model_path)

        # Pre-stack embeddings for fast similarity computation
        self.city_embeddings = np.vstack(self.city_df["embedding"].values)

    def recommend(self, query, n_cities=3, n_places_each=5, show_desc=True):

        query_vec = self.model.encode([query])

        # Find top matching cities using city’s embedding
        sims = cosine_similarity(query_vec, self.city_embeddings)[0]
        cities_copy = self.city_df.copy()
        cities_copy["similarity"] = sims

        # For each top city, find best places
        top_cities = (
            cities_copy.sort_values("similarity", ascending=False)
            .head(n_cities)
        )

        results = []

        for _, row in top_cities.iterrows():
            city_name = row["City"]

            # Get all places in that city
            subset = self.places_df[self.places_df["City"] == city_name].copy()
            if len(subset) == 0:
                continue

            place_vecs = np.vstack(subset["embedding"].values)
            # Compute similarity of each place to the query
            place_sims = cosine_similarity(query_vec, place_vecs)[0]

# calculate final score
# 70% semantic similarity ; 30% place rating (like review rating)
            subset["similarity"] = place_sims
            subset["final_score"] = (
                0.7 * subset["similarity"] +
                0.3 * (subset["Place_Rating"] / subset["Place_Rating"].max())
            )

            # select top places
            top_places = (
                subset.sort_values("final_score", ascending=False)
                .head(n_places_each)
            )

            for _, place in top_places.iterrows():
                results.append({
                    "City": city_name,
                    "Place": place["Place"],
                    "Rating": float(place["Place_Rating"]),
                    "Similarity": float(place["similarity"]),
                    "Description": place["Place_Desc"] if show_desc else None
                })

        return results
