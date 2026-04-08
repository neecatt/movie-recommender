# Date Movie Recommender

Couple-focused movie recommendation system built on the TMDB movie dataset.
Two people each choose one movie they already like, and the system returns one primary movie pick that both are likely to enjoy, plus a few backup options.

## Structure
- `notebooks/` for exploration and prototyping
- `src/` for reusable code
- `data/` for raw and processed datasets
- `models/` for saved models
- `reports/` for figures and evaluation outputs

## Product Flow
- Search and resolve an exact movie for person A
- Search and resolve an exact movie for person B
- Request one shared recommendation by `movie_id`
- Inspect the primary shared pick, explanations, and short fallback list

## Notebooks
- `notebooks/01_data_download.ipynb` downloads the dataset into `data/raw`
- `notebooks/02_eda.ipynb` performs preprocessing and EDA, then builds a TF-IDF baseline
