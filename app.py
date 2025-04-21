import requests
import aiohttp
import asyncio
from flask import Flask, jsonify, request
from flask_cors import CORS
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging
from functools import lru_cache, wraps
import random
import pandas as pd
import pickle
import time
import threading
import json
from pymongo import MongoClient
from datetime import datetime
import os
from model import main, get_recommendations
from transformers import pipeline
import nltk
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz
from dotenv import load_dotenv
import certifi
from chatbaot import search_movie_in_tmdb

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})
# Get FRONTEND_URLS from environment, split by comma, and strip whitespace
FRONTEND_URLS = os.getenv("FRONTEND_URLS", "http://localhost:3000").split(",")
FRONTEND_URLS = [url.strip() for url in FRONTEND_URLS if url.strip()]
logger.info(f"Allowed CORS origins: {FRONTEND_URLS}")

app = Flask(__name__)
# Configure CORS dynamically
CORS(app, resources={r"/api/*": {"origins": FRONTEND_URLS}})

# Download NLTK data
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

# Load NLP models
classifier = pipeline("zero-shot-classification")
ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# MongoDB Atlas setup
MONGO_URI = os.getenv("MONGO_URI")
try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tls=True,
        tlsAllowInvalidCertificates=True  # Temporary workaround for SSL issue
        # tlsCAFile=certifi.where()  # Re-enable once SSL issue is resolved
    )
    client.admin.command('ping')
    logger.info("Connected to MongoDB Atlas")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB Atlas: {e}")
    raise

db = client["movie_recommendation_db"]
users_collection = db["users"]
preferences_collection = db["user_preferences"]
movies_collection = db["movies_collection"]

# Create index for faster queries
preferences_collection.create_index("user_id", unique=True)
logger.info("Created index on user_id for preferences_collection")

# Initialize database with default admin user
def init_db():
    admin_user = {
        "id": 1,
        "username": "admin",
        "password": "admin123",
        "role": "admin",
        "last_login": None
    }
    if not users_collection.find_one({"username": "admin"}):
        users_collection.insert_one(admin_user)
        logger.info("Default admin user created.")
    else:
        users_collection.update_one(
            {"username": "admin"},
            {"$set": {"id": 1}},
            upsert=True
        )
        logger.info("Ensured admin user has an id field.")

init_db()

# Load data for content-based filtering
try:
    movies, cosine_sim = main()
except Exception as e:
    logger.error(f"Failed to load movie data: {e}")
    raise

# Load ratings for collaborative filtering
ratings_df = pd.read_csv(
    "dataset/dataset2/ratings.csv",
    dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int}
)

# Load links.csv for MovieLens to TMDB mapping
links_df = pd.read_csv(
    "dataset/dataset2/links.csv",
    dtype={"movieId": int, "imdbId": str, "tmdbId": str}
)

movielens_to_tmdb = dict(zip(links_df["movieId"], links_df["tmdbId"].astype(str)))

# Load collaborative filtering model
with open("svd_model.pkl", "rb") as file:
    collab_model = pickle.load(file)

# TMDB API Key
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "5bd7d31b6e1466d5799253aa07b28a02")

# Load or initialize poster cache
POSTER_CACHE_FILE = "poster_cache.json"
if os.path.exists(POSTER_CACHE_FILE):
    with open(POSTER_CACHE_FILE, "r") as f:
        poster_cache = json.load(f)
else:
    poster_cache = {}

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

MOVIE_TITLES = movies["title"].tolist()

# Global lock for TMDB requests
tmdb_lock = threading.Lock()

# Rate limiting decorator
def rate_limit(max_per_second):
    min_interval = 1.0 / float(max_per_second)
    def decorator(f):
        last_called = [0.0]
        @wraps(f)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                logger.info(f"Rate limiting: Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
            result = f(*args, **kwargs)
            last_called[0] = time.time()
            return result
        return wrapper
    return decorator

@rate_limit(max_per_second=3)
@lru_cache(maxsize=1000)
def search_tmdb_movie(title):
    with tmdb_lock:
        try:
            url = "https://api.themoviedb.org/3/search/movie"
            params = {
                "api_key": TMDB_API_KEY,
                "query": title,
                "language": "en-US",
                "page": 1,
            }
            logger.info(f"Searching TMDB for movie: {title}")
            response = session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data["results"]:
                tmdb_id = data["results"][0]["id"]
                logger.info(f"Found TMDB ID {tmdb_id} for: {title}")
                return tmdb_id
            logger.warning(f"No TMDB ID found for: {title}")
            return None
        except requests.RequestException as e:
            logger.error(f"Error searching TMDB for {title}: {e}")
            return None

async def fetch_poster_async(session, movie_id):
    if not movie_id:
        return movie_id, None
    movie_id_str = str(movie_id)
    if movie_id_str in poster_cache:
        return movie_id, poster_cache[movie_id_str]
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {"api_key": TMDB_API_KEY}
        async with session.get(url, params=params, timeout=15) as response:
            response.raise_for_status()
            data = await response.json()
            if "poster_path" in data and data["poster_path"]:
                poster_path = data["poster_path"]
                full_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
                poster_cache[movie_id_str] = full_path
                return movie_id, full_path
            return movie_id, None
    except Exception as e:
        logger.error(f"Error fetching poster for {movie_id}: {e}")
        return movie_id, None

def fetch_posters_batch(movie_ids):
    async def fetch_all():
        async with aiohttp.ClientSession() as session:
            tasks = [fetch_poster_async(session, mid) for mid in movie_ids]
            return await asyncio.gather(*tasks)
    results = asyncio.run(fetch_all())
    with open(POSTER_CACHE_FILE, "w") as f:
        json.dump(poster_cache, f)
    return dict(results)

def map_movielens_to_tmdb(movielens_movie_id):
    try:
        tmdb_id = movielens_to_tmdb.get(int(movielens_movie_id))
        if tmdb_id and tmdb_id != "nan":
            return int(tmdb_id)
        return None
    except Exception as e:
        logger.error(f"Error mapping MovieLens ID {movielens_movie_id}: {e}")
        return None

# Add index for movies_collection (add this after MongoDB setup in app.py)
movies_collection.create_index("movie_id", unique=True)
logger.info("Created index on movie_id for movies_collection")

def get_popular_movies(n=15):
    movie_ratings = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    popular_movies = movie_ratings[movie_ratings["count"] > 50].sort_values("mean", ascending=False)
    top_movie_ids = popular_movies["movieId"].head(n).tolist()
    recommendations = []
    tmdb_ids = [map_movielens_to_tmdb(mid) for mid in top_movie_ids]
    tmdb_ids = [tid for tid in tmdb_ids if tid]
    poster_urls = fetch_posters_batch(tmdb_ids)

    # Batch query movies_collection for trailer URLs
    trailer_docs = movies_collection.find({"movie_id": {"$in": tmdb_ids}}, {"movie_id": 1, "trailer_url": 1})
    trailer_map = {doc["movie_id"]: doc.get("trailer_url", "Trailer unavailable") for doc in trailer_docs}

    for movie_id in top_movie_ids:
        tmdb_id = map_movielens_to_tmdb(movie_id)
        if tmdb_id:
            movie_row = movies[movies["movie_id"] == tmdb_id]
            if not movie_row.empty:
                movie_title = movie_row["title"].values[0]
                poster_url = poster_urls.get(tmdb_id, "Poster unavailable")
                trailer_url = trailer_map.get(tmdb_id, "Trailer unavailable")
               # Extract genres
                genres = movie_row["genres"].values[0] if "genres" in movie_row and isinstance(movie_row["genres"].values[0], list) else ["No genres available"]
                # Extract year from release_date
                release_date = movie_row.get("release_date", "").values[0] if "release_date" in movie_row else ""
                year = release_date.split("-")[0] if release_date and "-" in release_date else ""
                
                recommendations.append({
                    "title": movie_title,
                    "movie_id": int(tmdb_id),
                    "predicted_rating": round(popular_movies[popular_movies["movieId"] == movie_id]["mean"].iloc[0], 2),
                    "poster": poster_url,
                    "trailer_url": trailer_url,
                    "genres": genres,
                    "year": year
                })
    logger.info(f"Popular movies: {recommendations}")
    return recommendations

# Cache user preferences
preferences_cache = {}

def get_content_based_fallback(user_id, n=10):
    if user_id in preferences_cache:
        user_prefs = preferences_cache[user_id]
    else:
        user_prefs = preferences_collection.find_one({"user_id": user_id})
        preferences_cache[user_id] = user_prefs

    if not user_prefs:
        logger.info(f"No preferences for user {user_id}, using popular movies")
        return get_popular_movies(n)

    user_genres = set(genre.lower() for genre in user_prefs.get("genres", []))
    user_actors = set(actor.lower() for actor in user_prefs.get("actors", []))
    user_directors = set(director.lower() for director in user_prefs.get("directors", []))

    if not (user_genres or user_actors or user_directors):
        logger.info(f"No specific preferences for user {user_id}, using popular movies")
        return get_popular_movies(n)

    # Vectorized scoring
    movies["score"] = 0
    movies["score"] += movies["genres"].apply(
        lambda x: sum(3 for g in user_genres if g in [genre.lower() for genre in x])
    )
    movies["score"] += movies.get("cast", []).apply(
        lambda x: sum(2 for a in user_actors if a in [actor.lower() for actor in x])
    )
    movies["score"] += movies.get("directors", []).apply(
        lambda x: sum(2 for d in user_directors if d in [director.lower() for director in x])
    )

    top_movies = movies[movies["score"] > 0][["title", "movie_id", "score"]].nlargest(n, "score")

    recommendations = []
    tmdb_ids = top_movies["movie_id"].tolist()
    poster_urls = fetch_posters_batch(tmdb_ids)

    for _, row in top_movies.iterrows():
        poster_url = poster_urls.get(row["movie_id"], "Poster unavailable")
        recommendations.append({
            "title": row["title"],
            "movie_id": int(row["movie_id"]),
            "predicted_rating": round(row["score"], 2),
            "poster": poster_url,
        })
    logger.info(f"Content-based fallback for user {user_id}: {recommendations}")
    return recommendations

def get_collaborative_recommendations(user_id, n=10):
    try:
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        if len(user_ratings) >= 5:
            logger.info(f"User {user_id} has {len(user_ratings)} ratings in ratings_df")
            all_movie_ids = ratings_df["movieId"].unique()
            rated_movie_ids = set(user_ratings["movieId"].values)
            predictions = []
            for movie_id in all_movie_ids:
                if movie_id not in rated_movie_ids:
                    predicted_rating = collab_model.predict(user_id, movie_id).est
                    predictions.append((movie_id, predicted_rating))
            predictions.sort(key=lambda x: x[1], reverse=True)
            top_movies = predictions[:n]
        else:
            logger.info(f"User {user_id} has insufficient ratings, checking MongoDB preferences")
            if user_id in preferences_cache:
                user_prefs = preferences_cache[user_id]
            else:
                user_prefs = preferences_collection.find_one({"user_id": user_id})
                preferences_cache[user_id] = user_prefs

            if user_prefs and "ratings" in user_prefs and len(user_prefs["ratings"]) >= 5:
                user_ratings = user_prefs["ratings"]
                user_genres = [genre.lower() for genre in user_prefs.get("genres", [])]
                user_actors = [actor.lower() for actor in user_prefs.get("actors", [])]
                user_directors = [director.lower() for director in user_prefs.get("directors", [])]
                logger.info(f"User {user_id} ratings: {user_ratings}")

                user_ratings_dict = {rating["movieId"]: rating["rating"] for rating in user_ratings}
                rated_movie_ids = set(user_ratings_dict.keys())
                all_movie_ids = ratings_df["movieId"].unique()
                predictions = []

                for movie_id in rated_movie_ids:
                    if movie_id in all_movie_ids:
                        user_rating = user_ratings_dict[movie_id]
                        predictions.append((movie_id, user_rating))
                        logger.info(f"Using user rating for movie {movie_id}: {user_rating}")

                for movie_id in all_movie_ids:
                    if movie_id not in rated_movie_ids:
                        predicted_rating = collab_model.predict(user_id, movie_id).est
                        tmdb_id = map_movielens_to_tmdb(movie_id)
                        if tmdb_id:
                            movie_row = movies[movies["movie_id"] == tmdb_id]
                            if not movie_row.empty:
                                movie_genres = [genre.lower() for genre in movie_row["genres"].iloc[0]]
                                movie_actors = [actor.lower() for actor in movie_row.get("cast", [])]
                                movie_directors = [director.lower() for director in movie_row.get("directors", [])]
                                genre_overlap = sum(1 for genre in user_genres if genre in movie_genres)
                                predicted_rating += genre_overlap * 1.0
                                actor_overlap = sum(1 for actor in user_actors if actor in movie_actors)
                                predicted_rating += actor_overlap * 0.5
                                director_overlap = sum(1 for director in user_directors if director in movie_directors)
                                predicted_rating += director_overlap * 0.5
                                if genre_overlap > 0 or actor_overlap > 0 or director_overlap > 0:
                                    logger.info(f"Boosted rating for movie {movie_id}: {predicted_rating}")
                        predictions.append((movie_id, predicted_rating))

                predictions.sort(key=lambda x: x[1], reverse=True)
                top_movies = predictions[:n]
            else:
                logger.info(f"Insufficient ratings for user {user_id}, using content-based")
                return get_content_based_fallback(user_id, n)

        recommendations = []
        tmdb_ids = [map_movielens_to_tmdb(movie_id) for movie_id, _ in top_movies]
        tmdb_ids = [tid for tid in tmdb_ids if tid]
        poster_urls = fetch_posters_batch(tmdb_ids)

        for movie_id, rating in top_movies:
            tmdb_id = map_movielens_to_tmdb(movie_id)
            if tmdb_id:
                movie_row = movies[movies["movie_id"] == tmdb_id]
                if not movie_row.empty:
                    movie_title = movie_row["title"].values[0]
                    poster_url = poster_urls.get(tmdb_id, "Poster unavailable")
                    recommendations.append({
                        "title": movie_title,
                        "movie_id": int(tmdb_id),
                        "predicted_rating": round(rating, 2),
                        "poster": poster_url,

                    })
        logger.info(f"Collaborative recommendations for user {user_id}: {recommendations}")
        return recommendations
    except Exception as e:
        logger.error(f"Error in collaborative filtering: {e}")
        return get_content_based_fallback(user_id, n)

@app.route("/api/user/preferences", methods=["POST"])
def save_user_preferences():
    try:
        data = request.json
        user_id = data.get("userId")
        if not user_id:
            return jsonify({"error": "userId is required"}), 400

        preferences = {
            "user_id": user_id,
            "genres": data.get("genres", []),
            "actors": data.get("actors", []) if data.get("actors") else [],
            "directors": data.get("directors", []) if data.get("directors") else [],
            "ratings": data.get("ratings", [])
        }

        preferences_collection.update_one(
            {"user_id": user_id},
            {"$set": preferences},
            upsert=True
        )
        preferences_cache[user_id] = preferences  # Update cache
        logger.info(f"Saved preferences for user {user_id}: {preferences}")
        return jsonify({"message": "Preferences saved successfully"})
    except Exception as e:
        logger.error(f"Error saving user preferences: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/user/preferences/<int:user_id>", methods=["GET"])
def get_user_preferences(user_id):
    try:
        if user_id in preferences_cache:
            prefs = preferences_cache[user_id]
        else:
            prefs = preferences_collection.find_one({"user_id": user_id}, {"_id": 0})
            preferences_cache[user_id] = prefs
        if prefs:
            return jsonify({"user_id": user_id, "preferences": prefs})
        return jsonify({"error": "User preferences not found"}), 404
    except Exception as e:
        logger.error(f"Error retrieving user preferences: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/clear_preferences/<int:user_id>", methods=["DELETE"])
def clear_preferences(user_id):
    try:
        result = preferences_collection.delete_one({"user_id": user_id})
        if result.deleted_count == 0:
            return jsonify({"error": "User preferences not found"}), 404
        preferences_cache.pop(user_id, None)  # Clear cache
        logger.info(f"Cleared preferences for user {user_id}")
        return jsonify({"message": "Preferences cleared"})
    except Exception as e:
        logger.error(f"Error clearing preferences: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/login", methods=["POST"])
def login():
    try:
        data = request.json
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return jsonify({"error": "Username and password are required"}), 400

        user = users_collection.find_one({"username": username, "password": password})
        if not user:
            return jsonify({"error": "Invalid username or password"}), 401

        users_collection.update_one(
            {"username": username},
            {"$set": {"last_login": datetime.utcnow().isoformat()}}
        )

        user_data = {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"],
            "last_login": user["last_login"]
        }

        logger.info(f"User logged in: username={username}, role={user['role']}")
        return jsonify({"message": "Login successful", "user": user_data})
    except Exception as e:
        logger.error(f"Error during login: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/users", methods=["GET"])
def get_users():
    try:
        users = list(users_collection.find({}, {"_id": 0}))
        return jsonify(users)
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user(user_id):
    try:
        user = users_collection.find_one({"id": user_id}, {"_id": 0})
        if user:
            return jsonify(user)
        else:
            abort(404, description="Resource not found")
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    try:
        result = users_collection.delete_one({"id": user_id})
        if result.deleted_count == 0:
            return jsonify({"error": "User not found"}), 404
        logger.info(f"Deleted user with ID {user_id}")
        return jsonify({"message": "User deleted successfully"})
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/register_user", methods=["POST"])
def register_user():
    try:
        data = request.json
        logger.info(f"Received registration request: {data}")

        highest_user = users_collection.find_one(sort=[("id", -1)])
        highest_id = highest_user.get("id", 0) if highest_user else 0
        next_user_id = highest_id + 1

        user_id = int(data.get("userId", next_user_id))
        username = data.get("username", f"user_{user_id}")
        password = data.get("password", "default_password_123")
        role = data.get("role", "user")

        logger.info(f"User data: userId={user_id}, username={username}, role={role}")

        if not username or username == f"user_{user_id}":
            logger.warning("Validation failed: Username is required")
            return jsonify({"error": "Username is required and cannot be default"}), 400
        if not password or password == "default_password_123":
            logger.warning("Validation failed: Password is required")
            return jsonify({"error": "Password is required and cannot be default"}), 400

        existing_user = users_collection.find_one({"username": username})
        if existing_user:
            logger.warning(f"Username already exists: {username}")
            return jsonify({"error": "Username already exists"}), 400

        new_user = {
            "id": user_id,
            "username": username,
            "password": password,
            "role": role,
            "last_login": None
        }

        users_collection.insert_one(new_user)
        logger.info(f"Registered new user: userId={user_id}, username={username}, role={role}")
        return jsonify({"message": "User registered successfully"})
    except Exception as e:
        logger.error(f"Error registering user: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/collaborative_recommendations/<int:user_id>", methods=["GET"])
def collaborative_recommend(user_id):
    recommendations = get_collaborative_recommendations(user_id)
    if recommendations:
        return jsonify({"user_id": user_id, "recommendations": recommendations})
    else:
        return jsonify({"error": "Could not generate recommendations"}), 404

@app.route("/api/movies", methods=["GET"])
def get_movies():
    try:
        page = int(request.args.get("page", 1))
        page_size = int(request.args.get("page_size", 10))

        if page < 1 or page_size < 1:
            return jsonify({"error": "Page and page_size must be positive integers."}), 400

        movies_cleaned = movies.copy()
        movies_cleaned['overview'] = movies_cleaned['overview'].fillna('No overview available')
        movies_cleaned['tags'] = movies_cleaned['tags'].fillna('')
        movies_cleaned = movies_cleaned.dropna(subset=['title'])

        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        total_movies = len(movies_cleaned)

        if start_idx >= total_movies:
            return jsonify({"error": "Page number exceeds available data."}), 404

        paginated_movies = movies_cleaned.iloc[start_idx:end_idx]
        tmdb_ids = paginated_movies["movie_id"].tolist()
        poster_urls = fetch_posters_batch(tmdb_ids)

        movies_with_posters = []
        for _, movie in paginated_movies.iterrows():
            movie_data = movie.to_dict()
            poster_url = poster_urls.get(movie['movie_id'], "Poster unavailable")
            movie_data['poster'] = poster_url
            movies_with_posters.append(movie_data)

        response_data = {
            "movies": movies_with_posters,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_movies": total_movies,
                "total_pages": (total_movies + page_size - 1) // page_size
            }
        }

        return jsonify(response_data)

    except ValueError:
        return jsonify({"error": "Invalid page or page_size parameters."}), 400
    except Exception as e:
        logger.error(f"Error processing movies request: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/api/recommendations/<int:movie_id>', methods=['GET'])
def recommendations(movie_id):
    try:
        movie_row = movies[movies['movie_id'] == movie_id]

        if movie_row.empty:
            return jsonify({"error": "Movie not found"}), 404

        movie_title = movie_row['title'].iloc[0]

        recommended_titles = get_recommendations(movie_title, cosine_sim, movies)

        recommended_titles = recommended_titles.tolist() if isinstance(recommended_titles, pd.Series) else recommended_titles
        if not recommended_titles:
            return jsonify({"error": "Could not generate recommendations"}), 404

        recommended_movies = []
        tmdb_ids = []
        for title in recommended_titles:
            movie_data = movies[movies['title'] == title].iloc[0].to_dict()
            tmdb_ids.append(movie_data['movie_id'])
            recommended_movies.append(movie_data)

        poster_urls = fetch_posters_batch(tmdb_ids)
        for movie_data in recommended_movies:
            movie_data['poster'] = poster_urls.get(movie_data['movie_id'], "Poster unavailable")

        response_data = {
            "movie_id": movie_id,
            "movie_title": movie_title,
            "recommendations": recommended_movies
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    try:
        movie = movies[movies['movie_id'] == movie_id]

        if movie.empty:
            return jsonify({"error": "Movie not found"}), 404

        movie_data = movie.iloc[0].to_dict()
        poster_urls = fetch_posters_batch([movie_data['movie_id']])
        movie_data['poster'] = poster_urls.get(movie_data['movie_id'], "Poster unavailable")

        return jsonify(movie_data)

    except Exception as e:
        logger.error(f"Error processing movie request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/hybrid_recommendations/<int:user_id>/<int:movie_id>', methods=['GET'])
def hybrid_recommend(user_id, movie_id):
    collab_recommendations = get_collaborative_recommendations(user_id, n=10)

    try:
        movie_row = movies[movies['movie_id'] == movie_id]
        if movie_row.empty:
            return jsonify({"error": f"Movie ID {movie_id} not found in content-based dataset"}), 404

        movie_title = movie_row['title'].iloc[0]

        recommended_titles = get_recommendations(movie_title, cosine_sim, movies)[:10]

        content_recommendations = []
        tmdb_ids = []
        for title in recommended_titles:
            movie_data = movies[movies['title'] == title].iloc[0].to_dict()
            tmdb_ids.append(movie_data['movie_id'])
            content_recommendations.append({
                "title": movie_data['title'],
                "movie_id": int(movie_data['movie_id']),
                "genres": movie_data.get('genres', []),
                "overview": movie_data.get('overview', "No overview available"),
            })

        poster_urls = fetch_posters_batch(tmdb_ids)
        for rec in content_recommendations:
            rec['poster'] = poster_urls.get(rec['movie_id'], "Poster unavailable")

        response_data = {
            "user_id": user_id,
            "movie_id": movie_id,
            "movie_title": movie_title,
            "hybrid_recommendations": {
                "collaborative": collab_recommendations,
                "content_based": content_recommendations
            }
        }
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Error in hybrid recommendation: {e}")
        return jsonify({"error": "Could not generate hybrid recommendations"}), 500

@app.route('/api/initial_recommendations', methods=['GET'])
def initial_recommendations():
    try:
        recommendations = get_popular_movies(n=20)
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        logger.error(f"Error fetching initial recommendations: {e}")
        return jsonify({"error": "Could not fetch initial recommendations"}), 500

# Chatbot Functions
def extract_genre(message):
    genres = [
        "action", "adventure", "animation", "comedy", "crime", "documentary",
        "drama", "family", "fantasy", "history", "horror", "music", "mystery",
        "romance", "science fiction", "sci-fi", "thriller", "war", "western",
    ]
    found_genres = []
    message_lower = message.lower()
    for genre in genres:
        if genre in message_lower:
            found_genres.append(genre)
    return found_genres

def get_movies_by_genre(genre, movies_df, count=5):
    try:
        genre = genre.lower()
        filtered_movies = []
        for _, movie in movies_df.iterrows():
            if isinstance(movie["genres"], list):
                movie_genres = [g.lower() if isinstance(g, str) else "" for g in movie["genres"]]
                if genre in movie_genres:
                    filtered_movies.append(movie)
            elif isinstance(movie["genres"], str):
                if genre.lower() in movie["genres"].lower():
                    filtered_movies.append(movie)

        if filtered_movies:
            selected_movies = random.sample(filtered_movies, min(count, len(filtered_movies)))
            recommendations = []
            tmdb_ids = [movie['movie_id'] for movie in selected_movies]
            poster_urls = fetch_posters_batch(tmdb_ids)
            for movie in selected_movies:
                poster_url = poster_urls.get(movie['movie_id'], "Poster unavailable")
                recommendations.append({
                    "title": movie["title"],
                    "movie_id": int(movie["movie_id"]),
                    "poster": poster_url,
                })
            return recommendations
        return []
    except Exception as e:
        logger.error(f"Error getting movies by genre: {e}")
        return []

def get_recommendations_from_input(message, movies_df, cosine_sim, count=5):
    try:
        genres = extract_genre(message)
        if genres:
            return get_movies_by_genre(genres[0], movies_df, count)

        random_movies = movies_df.sample(count)
        recommendations = []
        tmdb_ids = random_movies["movie_id"].tolist()
        poster_urls = fetch_posters_batch(tmdb_ids)
        for _, movie in random_movies.iterrows():
            poster_url = poster_urls.get(movie['movie_id'], "Poster unavailable")
            recommendations.append({
                'title': movie['title'],
                'movie_id': int(movie['movie_id']),
                'poster': poster_url,
            })
        return recommendations
    except Exception as e:
        logger.error(f"Error getting recommendations from input: {e}")
        return []

def detect_intent(user_message):
    labels = {
        "recommend_movies": "User wants movie recommendations",
        "find_similar": "User is asking for movies similar to another movie",
        "get_movie_details": "User wants information about a specific movie",
        "extract_genre": "User wants movie recommendations based on genre",
    }
    result = classifier(user_message, list(labels.values()))
    best_match = result["labels"][0]
    for key, value in labels.items():
        if value == best_match:
            return key
    return "unknown"

def extract_entities(user_message):
    entities = {"MOVIE": []}
    results = ner_model(user_message)

    current_entity = ""
    movie_candidates = []

    for res in results:
        entity_text = res["word"].replace("##", "")
        entity_label = res["entity"]
        if entity_label.startswith("B-") or entity_label.startswith("I-"):
            if res["word"].startswith("##"):
                current_entity += entity_text
            else:
                if current_entity:
                    movie_candidates.append(current_entity)
                current_entity = entity_text

    if current_entity:
        movie_candidates.append(current_entity)

    for movie in movie_candidates:
        fuzzy_result = process.extractOne(
            movie.lower(),
            [title.lower() for title in MOVIE_TITLES],
            scorer=fuzz.ratio,
            score_cutoff=75,
        )
        if fuzzy_result:
            match = fuzzy_result[0]
            validated_movie = search_tmdb_movie(match) if match else None
            if validated_movie:
                movie_row = movies[movies['movie_id'] == validated_movie]
                if movie_row.empty:
                    continue
                entities["MOVIE"].append(validated_movie)

    if len(entities["MOVIE"]) == 1:
        entities["MOVIE"] = entities["MOVIE"][0]

    return entities

def get_movie_details(movie_id_or_title):
    try:
        if isinstance(movie_id_or_title, (int, str)) and str(movie_id_or_title).isdigit():
            movie = movies[movies['movie_id'] == int(movie_id_or_title)]
            if movie.empty:
                return f"Sorry, I couldn't find details for the movie with ID {movie_id_or_title}.", None
        else:
            movie = movies[movies['title'].str.lower() == movie_id_or_title.lower()]
            if movie.empty:
                return f"Sorry, I couldn't find details for '{movie_id_or_title}'.", None

        movie_data = movie.iloc[0].to_dict()
        poster_urls = fetch_posters_batch([movie_data['movie_id']])
        overview = movie_data.get('overview', "No overview available.")
        return (
            f"Here's what I know about '{movie_data['title']}': {overview}",
            {
                "title": movie_data["title"],
                "movie_id": int(movie_data["movie_id"]),
                "overview": overview,
                "poster": poster_urls.get(movie_data['movie_id'], "Poster unavailable"),
            },
        )
    except Exception as e:
        logger.error(f"Error getting movie details: {e}")
        return f"Error retrieving movie details.", None

def format_recommendations(recommendations):
    if recommendations:
        movie_titles = [movie["title"] for movie in recommendations]
        return f"Here are some movies you might enjoy: {', '.join(movie_titles)}"
    return "I couldn't find any recommendations."

def format_genre_movies(genre_movies, genre):
    if genre_movies:
        movie_titles = [movie["title"] for movie in genre_movies]
        return f"Here are some {genre} movies: {', '.join(movie_titles)}"
    return f"No movies found for genre '{genre}'."

def format_similar_movies(similar_movies, movie_title):
    if similar_movies:
        movie_titles = [movie["title"] for movie in similar_movies]
        return f"If you liked {movie_title}, you might also enjoy: {', '.join(movie_titles)}"
    return f"Couldn't find similar movies for '{movie_title}'."

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        action = detect_intent(user_message)
        entities = extract_entities(user_message)
        movie_id_or_title = entities.get("MOVIE")
        genre = extract_genre(user_message)

        if action == 'recommend_movies':
            recommendations = get_recommendations_from_input(user_message, movies, cosine_sim)
            response_text = format_recommendations(recommendations)
            action_result = recommendations

        elif action == 'extract_genre' and genre:
            genre_movies = get_movies_by_genre(genre[0], movies)
            response_text = format_genre_movies(genre_movies, genre[0])
            action_result = genre_movies

        elif action == 'get_movie_details' and movie_id_or_title:
            response_text, action_result = get_movie_details(movie_id_or_title)

        elif action == 'find_similar' and movie_id_or_title:
            similar_movies = get_recommendations_from_input(f"like {movie_id_or_title}", movies, cosine_sim)
            response_text = format_similar_movies(similar_movies, movie_id_or_title)
            action_result = similar_movies

        else:
            response_text = "I'm not sure what you're asking. Try asking for movie recommendations or details about a specific movie!"
            action_result = []

        return jsonify({"response": response_text, "action": action, "result": action_result})

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    print("\n🚀 Available API Routes:")
    for rule in app.url_map.iter_rules():
        print(rule)
    print("\n")
    app.run(debug=False, host=host, port=port)