import requests


TMDB_API_KEY = '5bd7d31b6e1466d5799253aa07b28a02'

import time
import requests

def search_movie_in_tmdb(movie_name, max_retries=10):
    """Search for a movie in the TMDB API with retries."""
    base_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_name
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, params=params, timeout=5)
            response.raise_for_status()  # Raise error for HTTP issues

            data = response.json()
            results = data.get("results", [])

            if results:
                return results[0]['title']  # Return the best match
            else:
                print(f"⚠️ No results found in TMDB for: {movie_name}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"⚠️ TMDB API request failed ({e}). Retrying in {2**attempt} sec...")
            time.sleep(2 ** attempt)

    print(f"❌ Failed to retrieve {movie_name} after {max_retries} attempts.")
    return None