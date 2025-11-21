# 🌍 Travel Recommender API (FastAPI + SentenceTransformer)

A smart **travel destination recommendation API** built using **FastAPI**, **SentenceTransformer embeddings**, and a **hybrid semantic scoring approach**.  
This API takes a user’s travel intent (e.g., *“relaxing beach vacation, water sports, tropical vibes”*) and returns the best matching places ranked by relevance.

This project is cleanly structured, production-ready, and optimized for deployment on **Render**, **Railway**, or any cloud platform.

---
[![Watch the demo](https://img.shields.io/badge/😊-Watch%20Demo-red)](https://vimeo.com/1139192676?fl=ip&fe=ec) 

## ✨ Features

- 🔍 **Semantic query understanding** using `all-MiniLM-L6-v2`
- 🧠 **Hybrid scoring**: combines semantic similarity + place ratings
- 🚀 **FastAPI backend** for high-performance inference
- 📦 **Precomputed embeddings** for instant response times
- 📘 **Automatic Swagger UI** for testing (`/docs`)
- 🔒 Ready for API key auth & CORS (optional)
- ☁️ Model + embeddings fully saved for cloud deployment

---
⚙️ Hybrid Recommendation Logic

The recommender uses:

1. City-level semantic matching

Embed user query

Compare with each city’s embedding

Select top N cities

2. Place-level re-ranking

Each place inside selected cities gets a score:

final_score = (0.7 × semantic_similarity) + 
              (0.3 × normalized_place_rating)

3. Top places returned as JSON

This approach produces highly relevant, context-aware recommendations.

---
🚀 Run the API

Start the FastAPI server using Uvicorn:
uvicorn main:app --reload

Your API will run at:
http://127.0.0.1:8000

📘 Interactive Swagger Docs

Open:

http://127.0.0.1:8000/docs

You can test the /recommend endpoint directly there.
