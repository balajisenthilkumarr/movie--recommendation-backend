import pandas as pd
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict

# Load ratings data
ratings = pd.read_csv(
    "dataset/dataset2/ratings.csv",
    sep=",",
    names=["userId", "movieId", "rating", "timestamp"],
    header=0,  # First row is the header
    dtype={"userId": int, "movieId": int, "rating": float, "timestamp": int},
    engine="python",
    encoding="latin1"
)

# Load movies data
movies_df = pd.read_csv(
    "dataset/dataset2/movies.csv",
    sep=",",
    names=["movieId", "title", "genres"],
    header=0,  # First row is the header
    dtype={"movieId": int, "title": str, "genres": str},
    engine="python",
    encoding="latin1"
)

print("✅ Successfully loaded ratings and movies data!")
print(f"Total ratings: {len(ratings)}")

# Drop timestamps since they are not needed
ratings.drop(columns=["timestamp"], inplace=True)

# Sample the dataset to reduce training time (optional, comment out if you want to use the full dataset)
SAMPLE_SIZE = 1_000_000  # Use 1 million ratings for training
if len(ratings) > SAMPLE_SIZE:
    ratings = ratings.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"✅ Sampled dataset to {SAMPLE_SIZE} ratings for faster training.")
else:
    print(f"✅ Using full dataset with {len(ratings)} ratings!")

# Define Reader for Surprise Library
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)

# Split the dataset into train and test sets (80% train, 20% test)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Train SVD model
print("Training model on sampled dataset...")
model = SVD(n_epochs=20)  # Increase epochs for better training
model.fit(trainset)
print("✅ Training complete.")

# Predict on test data
predictions = model.test(testset)

# Check accuracy
rmse = accuracy.rmse(predictions)
print(f"📊 RMSE: {rmse}")

# Function to get top-N recommendations
def get_top_n_recommendations(predictions, n=10):
    """Return the top-N recommendations for each user."""
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# Get recommendations
top_n_recommendations = get_top_n_recommendations(predictions)

# Save trained model using pickle
with open("svd_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model saved as 'svd_model.pkl'.")

# Print recommendations for multiple users
for user_id, movies in list(top_n_recommendations.items())[:5]:  # Show top 5 users
    print(f"\n🎬 Top recommendations for User {user_id}:")
    for movie_id, rating in movies:
        movie_title = movies_df[movies_df["movieId"] == movie_id]["title"].values
        movie_title = movie_title[0] if len(movie_title) > 0 else "Unknown Movie"
        print(f"📌 {movie_title} (Predicted Rating: {rating:.2f})")