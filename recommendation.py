import pandas as pd
import requests
import nltk
import os
import ast
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


API_KEY = "73f2c58b"

ps = PorterStemmer()
def convert(obj):

    genres = []

    try:

        for item in ast.literal_eval(obj):

            genres.append(item['name'])

    except:

        pass

    return ", ".join(genres)


def stem(text):

    y = []

    for word in text.split():

        y.append(ps.stem(word))

    return " ".join(y)


def fetch_movie_details(movie_title):

    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={API_KEY}"

    response = requests.get(url)

    data = response.json()

    poster = data.get("Poster")

    if poster == "N/A" or not poster:

        poster = "https://via.placeholder.com/300x450?text=No+Image"

    return {
        "title": data.get("Title", movie_title),
        "poster": poster,
        "rating": data.get("imdbRating", "N/A")
    }


movies = pd.read_csv('movies.csv')
movies = movies.head(2000)

movies = movies[
    ['title', 'overview', 'genres']
]

movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies['tags'] = (
    movies['overview'].astype(str) + ' ' +
    movies['genres'].astype(str)
)

movies['tags'] = movies['tags'].apply(lambda x: x.lower())

movies['tags'] = movies['tags'].apply(stem)

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = vectorizer.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)


def recommend(movie_name):

    movie_name = movie_name.lower()

    movie_index = None

    for index, movie in enumerate(movies['title']):

        if movie_name in str(movie).lower():

            movie_index = index
            break

    if movie_index is None:
        return []

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:9]

    recommendations = []

    for movie in movies_list:

        movie_title = movies.iloc[movie[0]]['title']
        movie_genres = movies.iloc[movie[0]]['genres']

        movie_data = fetch_movie_details(movie_title)
        movie_data["explanation"] = (
           f"Recommended because it shares "
           f"{movie_genres} themes."
        )

        movie_genres = (
    movies.iloc[movie[0]]['genres']
    .replace('|', ', ')
)

        movie_data["explanation"] = (
        f"Recommended because it shares "
        f"{movie_genres} themes."
)

        recommendations.append(movie_data)

    return recommendations


def recommend_by_mood(mood):

    mood_movies = movies[
        movies['genres'].str.contains(
            mood,
            case=False,
            na=False
        )
    ]

    recommendations = []

    for movie in mood_movies['title'].head(5):

        movie_data = fetch_movie_details(movie)

    movie_genres = (
       movies.iloc[movie[0]]['genres']
       .replace('|', ', ')
)

    movie_data["explanation"] = (
       f"Recommended because it shares "
       f"{movie_genres} themes."
)

    recommendations.append(movie_data)

    return recommendations