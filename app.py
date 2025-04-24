import streamlit as st
import pandas as pd
from recommendation import recommend_movies

# ✅ Load Movies Data
movies_df = pd.read_csv("movies.csv")

# ✅ Streamlit UI
st.title("🎬 Movie Recommendation System")
st.write("Enter a movie title to get similar recommendations.")

# ✅ Dropdown for movie selection
movie_title_input = st.selectbox("Select a movie title:", movies_df['title'])

if movie_title_input:
    recommendations = recommend_movies(movie_title_input)
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.write("Recommended Movies:")
        for idx, title in enumerate(recommendations):
            st.write(f"{idx + 1}. {title}")
