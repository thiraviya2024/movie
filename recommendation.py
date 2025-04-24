import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Load Movies Dataset
movies_path = "movies.csv"
if not os.path.exists(movies_path):
    print("Error: movies.csv not found!")
    exit()

movies_df = pd.read_csv(movies_path)
movies_df.fillna({"listed_in": "", "description": "", "cast": "", "director": ""}, inplace=True)

# ✅ Create a Combined Feature Column
movies_df["combined_features"] = (
    movies_df["title"] + " " + movies_df["listed_in"] + " " +
    movies_df["description"] + " " + movies_df["cast"] + " " + movies_df["director"]
)

# ✅ Compute Similarity Matrix (Save for Faster Loading)
vectorizer = TfidfVectorizer(stop_words="english")
features_matrix = vectorizer.fit_transform(movies_df["combined_features"])

similarity_matrix_path = "similarity_matrix.pkl"
if not os.path.exists(similarity_matrix_path):
    with open(similarity_matrix_path, "wb") as f:
        pickle.dump(features_matrix, f)
else:
    with open(similarity_matrix_path, "rb") as f:
        features_matrix = pickle.load(f)

similarity_matrix = cosine_similarity(features_matrix)

# ✅ Movie Recommendation Function
def recommend_movies(movie_title):
    movie_title = movie_title.lower().strip()
    movie = movies_df[movies_df["title"].str.lower() == movie_title]

    if movie.empty:
        return "Movie not found! Try another title."

    idx = movie.index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

    return [movies_df.iloc[i[0]]["title"] for i in sim_scores]
