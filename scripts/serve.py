from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import time
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.models.hybrid import ArtifactCompatibilityError, HybridRecommender


def load_model() -> HybridRecommender:
    project_root = Path(__file__).resolve().parents[1]
    artifacts_path = project_root / "models" / "hybrid_artifacts.joblib"
    if not artifacts_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found: {artifacts_path}. Run scripts/train.py first."
        )
    artifacts: dict[str, Any] = joblib.load(artifacts_path)
    HybridRecommender.validate_artifacts(artifacts)
    return HybridRecommender.from_artifacts(artifacts, validate=False)


def create_app(model: HybridRecommender | None = None) -> FastAPI:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("movie_recommender")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.model is None:
            app.state.model = load_model()
        yield

    app = FastAPI(title="Date Movie Recommender", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.model = model

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

    def get_model() -> HybridRecommender:
        current = app.state.model
        if current is None:
            raise HTTPException(status_code=503, detail="Model is not loaded.")
        return current

    @app.get("/health")
    def health() -> dict[str, str]:
        try:
            get_model()
        except HTTPException:
            return {"status": "starting"}
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return """
        <html>
          <head>
            <title>Date Movie Recommender</title>
            <style>
              :root {
                --bg: #f5efe3;
                --panel: #fffaf1;
                --ink: #1f1b16;
                --muted: #6a5e50;
                --line: #d9ccb9;
                --accent: #c75b39;
                --accent-dark: #8e361e;
              }
              body {
                margin: 0;
                font-family: Georgia, "Times New Roman", serif;
                background:
                  radial-gradient(circle at top left, #fff8ee 0, #f5efe3 40%, #efe3d1 100%);
                color: var(--ink);
              }
              .page {
                max-width: 980px;
                margin: 0 auto;
                padding: 32px 20px 56px;
              }
              .hero {
                background: rgba(255, 250, 241, 0.88);
                border: 1px solid var(--line);
                border-radius: 24px;
                padding: 28px;
                box-shadow: 0 20px 50px rgba(77, 54, 35, 0.08);
              }
              h1 { margin: 0 0 10px; font-size: 2.6rem; }
              p { color: var(--muted); line-height: 1.55; }
              .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 18px;
                margin-top: 24px;
              }
              .picker, .result-card, .alt-card {
                background: var(--panel);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 18px;
              }
              input, select, button {
                width: 100%;
                padding: 12px 14px;
                border-radius: 12px;
                border: 1px solid var(--line);
                font: inherit;
                box-sizing: border-box;
              }
              button {
                background: linear-gradient(135deg, var(--accent), #df8c4c);
                color: white;
                border: none;
                font-weight: 700;
                cursor: pointer;
              }
              button:hover { background: linear-gradient(135deg, var(--accent-dark), var(--accent)); }
              .picker label, .result-section h2, .alt-list h3 {
                display: block;
                margin-bottom: 10px;
                font-weight: 700;
              }
              .picker small { color: var(--muted); display: block; margin-top: 8px; }
              .actions { margin-top: 18px; }
              .result-section { margin-top: 26px; display: none; }
              .badge {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 999px;
                background: #f3d7bf;
                color: var(--accent-dark);
                font-size: 0.85rem;
                margin-bottom: 12px;
              }
              .explain-group { margin-top: 14px; }
              .explain-group strong { display: block; margin-bottom: 6px; }
              .explain-group ul { margin: 0; padding-left: 18px; color: var(--muted); }
              .alt-list { margin-top: 18px; }
              .alt-card { margin-top: 12px; }
              .meta { color: var(--muted); margin-top: 6px; }
              .status { margin-top: 14px; color: var(--muted); min-height: 24px; }
            </style>
          </head>
          <body>
            <div class="page">
              <section class="hero">
                <span class="badge">Couple Movie Match</span>
                <h1>Pick one movie you both can enjoy.</h1>
                <p>Search one movie for each person, lock the exact titles, then get one primary shared recommendation plus a few backups.</p>
                <div class="grid">
                  <div class="picker">
                    <label for="queryA">Person A</label>
                    <input id="queryA" placeholder="Search a movie title" />
                    <div class="actions"><button type="button" onclick="searchMovies('A')">Search for Person A</button></div>
                    <select id="selectA" size="6"></select>
                    <small id="selectedA">No movie selected yet.</small>
                  </div>
                  <div class="picker">
                    <label for="queryB">Person B</label>
                    <input id="queryB" placeholder="Search a movie title" />
                    <div class="actions"><button type="button" onclick="searchMovies('B')">Search for Person B</button></div>
                    <select id="selectB" size="6"></select>
                    <small id="selectedB">No movie selected yet.</small>
                  </div>
                </div>
                <div class="actions"><button type="button" onclick="recommend()">Find Our Movie</button></div>
                <div class="status" id="status"></div>
              </section>

              <section class="result-section" id="results">
                <div class="result-card">
                  <span class="badge">Best Shared Pick</span>
                  <h2 id="bestTitle"></h2>
                  <div class="meta" id="bestMeta"></div>
                  <div class="explain-group">
                    <strong>Why it works for both</strong>
                    <ul id="worksForBoth"></ul>
                  </div>
                  <div class="explain-group">
                    <strong>Why it leans toward Person A</strong>
                    <ul id="leansA"></ul>
                  </div>
                  <div class="explain-group">
                    <strong>Why it leans toward Person B</strong>
                    <ul id="leansB"></ul>
                  </div>
                </div>
                <div class="alt-list">
                  <h3>Alternatives</h3>
                  <div id="alternatives"></div>
                </div>
              </section>
            </div>
            <script>
              const selections = { A: null, B: null };

              function optionLabel(movie) {
                const year = movie.year ? ` (${movie.year})` : '';
                return `${movie.movie_name}${year} • ${movie.genre || 'Unknown genre'}`;
              }

              function fillList(side, items) {
                const select = document.getElementById(`select${side}`);
                select.innerHTML = '';
                items.forEach((movie, idx) => {
                  const option = document.createElement('option');
                  option.value = movie.movie_id;
                  option.textContent = optionLabel(movie);
                  option.dataset.payload = JSON.stringify(movie);
                  if (idx === 0) option.selected = true;
                  select.appendChild(option);
                });
                if (items.length > 0) {
                  selections[side] = items[0];
                  document.getElementById(`selected${side}`).textContent = `Selected: ${optionLabel(items[0])}`;
                }
              }

              async function searchMovies(side) {
                const query = document.getElementById(`query${side}`).value.trim();
                const status = document.getElementById('status');
                if (!query) {
                  status.textContent = `Enter a title for person ${side}.`;
                  return;
                }
                status.textContent = 'Searching...';
                const resp = await fetch(`/search?query=${encodeURIComponent(query)}&limit=8`);
                const data = await resp.json();
                fillList(side, data.results || []);
                if (!data.results || data.results.length === 0) {
                  selections[side] = null;
                  document.getElementById(`selected${side}`).textContent = 'No matches found.';
                  status.textContent = 'No matches found. Try a more specific title.';
                  return;
                }
                status.textContent = `Found ${data.results.length} options for person ${side}.`;
              }

              ['A', 'B'].forEach((side) => {
                document.addEventListener('change', (event) => {
                  if (event.target.id !== `select${side}`) return;
                  const option = event.target.selectedOptions[0];
                  if (!option) return;
                  const movie = JSON.parse(option.dataset.payload);
                  selections[side] = movie;
                  document.getElementById(`selected${side}`).textContent = `Selected: ${optionLabel(movie)}`;
                });
              });

              function renderList(id, items) {
                const root = document.getElementById(id);
                root.innerHTML = '';
                const values = items && items.length ? items : ['No strong signal here.'];
                values.forEach((text) => {
                  const li = document.createElement('li');
                  li.textContent = text;
                  root.appendChild(li);
                });
              }

              async function recommend() {
                const status = document.getElementById('status');
                if (!selections.A || !selections.B) {
                  status.textContent = 'Select one exact movie for each person before requesting a match.';
                  return;
                }
                status.textContent = 'Finding your shared movie...';
                const resp = await fetch(`/recommend?movie_id_a=${encodeURIComponent(selections.A.movie_id)}&movie_id_b=${encodeURIComponent(selections.B.movie_id)}&alternatives_n=4`);
                const data = await resp.json();
                if (!resp.ok) {
                  status.textContent = data.detail || 'Recommendation failed.';
                  return;
                }
                document.getElementById('results').style.display = 'block';
                document.getElementById('bestTitle').textContent = data.best_pick.movie_name;
                document.getElementById('bestMeta').textContent =
                  `${data.best_pick.year || 'Unknown year'} • ${data.best_pick.genre || 'Unknown genre'} • Rating ${data.best_pick.rating ?? 'N/A'} • Shared fit ${data.best_pick.shared_fit_score.toFixed(3)}`;
                renderList('worksForBoth', data.explanation.works_for_both);
                renderList('leansA', data.explanation.leans_to_a);
                renderList('leansB', data.explanation.leans_to_b);

                const altRoot = document.getElementById('alternatives');
                altRoot.innerHTML = '';
                data.alternatives.forEach((movie) => {
                  const card = document.createElement('div');
                  card.className = 'alt-card';
                  card.innerHTML = `<strong>${movie.movie_name}</strong><div class="meta">${movie.year || 'Unknown year'} • ${movie.genre || 'Unknown genre'} • Shared fit ${movie.shared_fit_score.toFixed(3)}</div>`;
                  altRoot.appendChild(card);
                });
                status.textContent = `Best match selected for ${data.movie_a.movie_name} and ${data.movie_b.movie_name}.`;
              }
            </script>
          </body>
        </html>
        """

    @app.get("/search")
    def search(
        query: str = Query(..., min_length=1),
        limit: int = Query(8, ge=1, le=20),
    ) -> dict[str, Any]:
        results = get_model().search_movies(query, limit=limit)
        return {"query": query, "results": results}

    @app.get("/recommend")
    def recommend(
        movie_id_a: str = Query(...),
        movie_id_b: str = Query(...),
        alternatives_n: int = Query(4, ge=0, le=5),
        debug: bool = Query(False),
    ) -> dict[str, Any]:
        try:
            return get_model().recommend_date_movie(
                movie_id_a,
                movie_id_b,
                alternatives_n=alternatives_n,
                include_debug=debug,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ArtifactCompatibilityError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
