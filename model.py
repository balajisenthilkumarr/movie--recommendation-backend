import pandas as pd
import ast
import pickle
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    try:
        credits = pd.read_csv('./dataset/tmdb_5000_credits.csv')
        movies = pd.read_csv('./dataset/tmdb_5000_movies.csv')

        # Rename 'id' to 'movie_id' for consistency
        movies = movies.rename(columns={'id': 'movie_id'})

        # Log available columns
        logger.info(f"Movies columns: {movies.columns.tolist()}")
        logger.info(f"Credits columns: {credits.columns.tolist()}")

        # Merge on title
        movies = movies.merge(credits, left_on='title', right_on='title', how='left')

        # Handle movie_id conflict (select movie_id_x from movies)
        if 'movie_id_x' in movies.columns:
            movies['movie_id'] = movies['movie_id_x']
            movies = movies.drop(columns=['movie_id_x', 'movie_id_y'], errors='ignore')
        elif 'movie_id' not in movies.columns:
            logger.error("movie_id column missing after merge")
            raise ValueError("movie_id column missing after merge")

        # Select relevant columns
        movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'original_language', 'release_date']]

        # Verify release_date
        if 'release_date' not in movies.columns:
            logger.error("release_date column missing in dataset")
            raise ValueError("release_date column missing in dataset")

        logger.info(f"Missing release_date count: {movies['release_date'].isna().sum()}")
        logger.info(f"Sample release_date values: {movies['release_date'].dropna().head().tolist()}")

        return movies
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def convert(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except (ValueError, SyntaxError) as e:
        logger.warning(f"Error parsing object: {e}")
    return L

def process_data(movies):
    try:
        movies['genres'] = movies['genres'].apply(convert)
        movies['keywords'] = movies['keywords'].apply(convert)
        movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)[:3]] if pd.notna(x) else [])
        movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'] if pd.notna(x) else [])

        movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
        movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
        movies['tags'] = movies['tags'].apply(lambda x: x.lower())

        movies['release_date'] = movies['release_date'].fillna('')
        return movies
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

def compute_similarity(movies):
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['tags'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return cosine_sim
    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        raise

def precompute_recommendations(movies, cosine_sim):
    try:
        recommendation_cache = {}
        for idx, title in enumerate(movies['title']):
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            movie_indices = [i[0] for i in sim_scores]
            recommendation_cache[title] = movies['title'].iloc[movie_indices].tolist()
        
        with open('recommendation_cache.pkl', 'wb') as f:
            pickle.dump(recommendation_cache, f)
        return recommendation_cache
    except Exception as e:
        logger.error(f"Error precomputing recommendations: {e}")
        raise

def get_recommendations(title, cosine_sim=None, movies=None):
    try:
        if os.path.exists('recommendation_cache.pkl'):
            with open('recommendation_cache.pkl', 'rb') as f:
                recommendation_cache = pickle.load(f)
            if title in recommendation_cache:
                return recommendation_cache[title]
        
        idx = movies[movies['title'] == title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return movies['title'].iloc[movie_indices].tolist()
    except Exception as e:
        logger.error(f"Error getting recommendations for {title}: {e}")
        return []

def save_data(movies, cosine_sim):
    try:
        with open('movie_data.pkl', 'wb') as f:
            pickle.dump((movies, cosine_sim), f)
        logger.info("Saved movies and cosine_sim to movie_data.pkl")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def load_pickled_data():
    try:
        with open('movie_data.pkl', 'rb') as f:
            movies, cosine_sim = pickle.load(f)
        logger.info("Loaded movies and cosine_sim from movie_data.pkl")
        return movies, cosine_sim
    except Exception as e:
        logger.error(f"Error loading pickled data: {e}")
        raise

def main():
    try:
        movies, cosine_sim = load_pickled_data()
        if not os.path.exists('recommendation_cache.pkl'):
            precompute_recommendations(movies, cosine_sim)
    except (FileNotFoundError, Exception) as e:
        logger.info(f"Pickle file missing or corrupted, regenerating: {e}")
        movies = load_data()
        movies = process_data(movies)
        cosine_sim = compute_similarity(movies)
        save_data(movies, cosine_sim)
        precompute_recommendations(movies, cosine_sim)
    return movies, cosine_sim

if __name__ == '__main__':
    movies, cosine_sim = main()
    print(get_recommendations('The Dark Knight Rises', cosine_sim, movies))