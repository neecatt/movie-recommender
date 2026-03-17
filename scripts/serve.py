from __future__ import annotations

import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from src.models.hybrid import HybridRecommender


def load_model() -> HybridRecommender:
    project_root = Path(__file__).resolve().parents[1]
    artifacts_path = project_root / "models" / "hybrid_artifacts.joblib"
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found: {artifacts_path}. Run scripts/train.py first."
        )
    artifacts: dict[str, Any] = joblib.load(artifacts_path)
    return HybridRecommender.from_artifacts(artifacts)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("movie_recommender")

app = FastAPI(title="Movie Recommender")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = load_model()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration, 2),
        },
    )
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <html>
      <head><title>Movie Recommender</title></head>
      <body style="font-family: sans-serif; max-width: 720px; margin: 40px auto;">
        <h1>Movie Recommender</h1>
        <p>Enter two movie titles to get recommendations.</p>
        <input id="titleA" style="width: 100%; padding: 8px;" placeholder="Movie title A" />
        <input id="titleB" style="width: 100%; padding: 8px; margin-top: 8px;" placeholder="Movie title B" />
        <button onclick="getRecs()" style="margin-top: 12px;">Recommend</button>
        <pre id="output" style="margin-top: 20px; white-space: pre-wrap;"></pre>
        <script>
          async function getRecs() {
            const titleA = document.getElementById('titleA').value;
            const titleB = document.getElementById('titleB').value;
            const resp = await fetch(`/recommend?title_a=${encodeURIComponent(titleA)}&title_b=${encodeURIComponent(titleB)}&top_n=10`);
            const data = await resp.json();
            document.getElementById('output').textContent = JSON.stringify(data, null, 2);
          }
        </script>
      </body>
    </html>
    """


@lru_cache(maxsize=1024)
def _cached_recommend(title: str, top_n: int):
    return _model.recommend(title, top_n=top_n)


@app.get("/recommend")
def recommend(
    title_a: str = Query(...),
    title_b: str = Query(...),
    top_n: int = Query(10, ge=1, le=50),
) -> dict:
    try:
        results = _model.recommend_from_two(title_a, title_b, top_n=top_n)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "title_a": title_a,
        "title_b": title_b,
        "results": [
            {
                "movie_name": r.movie_name,
                "year": r.year,
                "rating": r.rating,
                "genre": r.genre,
                "score": r.score,
                "reasons": r.reasons,
            }
            for r in results
        ],
    }
