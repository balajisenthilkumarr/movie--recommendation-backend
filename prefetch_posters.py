

import json
import pandas as pd
import logging
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from model import main
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# TMDB API Key
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "5bd7d31b6e1466d5799253aa07b28a02")

# Poster cache file
POSTER_CACHE_FILE = "poster_cache.json"

# Configure session with retry strategy
def create_session():
    session = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

session = create_session()

def fetch_poster(movie_id):
    if not movie_id:
        logger.warning("No movie ID provided for poster.")
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY}
        logger.info(f"Fetching poster for movie ID {movie_id}")
        response = session.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if "poster_path" in data and data["poster_path"]:
            poster_path = data["poster_path"]
            full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
            logger.info(f"Cached poster for movie ID {movie_id}")
            return full_path
        logger.warning(f"No poster found for movie ID {movie_id}")
        return None
    except requests.RequestException as e:
        logger.error(f"Error fetching poster for {movie_id}: {e}")
        return None

def pre_fetch_posters(movies_df, cache_file=POSTER_CACHE_FILE):
    poster_cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            poster_cache = json.load(f)

    for _, movie in tqdm(movies_df.iterrows(), total=len(movies_df), desc="Fetching posters"):
        movie_id = str(movie["movie_id"])
        if movie_id not in poster_cache:
            poster_url = fetch_poster(movie_id)
            if poster_url:
                poster_cache[movie_id] = poster_url

    with open(cache_file, "w") as f:
        json.dump(poster_cache, f)
    logger.info(f"Updated poster cache with {len(poster_cache)} entries")

if __name__ == "__main__":
    movies, _ = main()
    pre_fetch_posters(movies)