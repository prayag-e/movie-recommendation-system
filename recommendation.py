import pandas as pd
import requests
import nltk
import os
import ast
import urllib3

urllib3.disable_warnings()
from nltk.stem.porter import PorterStemmer
from urllib.parse import quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TMDB_API_KEY = "77adb6aedb53d68ab4b08ff7cfbfd715"
movie_cache = {}
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
    if movie_title in movie_cache:
        return movie_cache[movie_title]
    search_url = (
        f"https://api.themoviedb.org/3/search/movie"
        f"?api_key={TMDB_API_KEY}"
        f"&query={quote(movie_title)}"
    )
    

    headers = {
    "User-Agent": "Mozilla/5.0"
}

    response = requests.get(
        search_url,
        headers=headers,
        timeout=10
)

    data = response.json()

    results = data.get("results")

    if not results:

        return {
            "title": movie_title,
            "poster": "https://via.placeholder.com/300x450?text=No+Image",
            "rating": "N/A",
            "genres": [],
            "explanation": "No description available."
        }

    movie = results[0]

    poster_path = movie.get("poster_path")

    if poster_path:

        poster = (
            "https://image.tmdb.org/t/p/w500"
            + poster_path
        )

    else:

        poster = (
            "https://via.placeholder.com/300x450?text=No+Image"
        )

    rating = movie.get("vote_average", "N/A")

    genre_ids = movie.get("genre_ids", [])

    genre_map = {

        28: "Action",
        12: "Adventure",
        16: "Animation",
        35: "Comedy",
        80: "Crime",
        18: "Drama",
        14: "Fantasy",
        27: "Horror",
        9648: "Mystery",
        10749: "Romance",
        878: "Science Fiction",
        53: "Thriller"

    }

    genres = [
        genre_map.get(gid)
        for gid in genre_ids
        if genre_map.get(gid)
    ]

    explanation = (
        "Recommended because it shares "
        + ", ".join(genres[:4])
        + " themes."
    )

    movie_cache[movie_title] = {

        "title": movie.get("title", movie_title),

        "poster": poster,

        "rating": round(rating, 1)
        if isinstance(rating, (int, float))
        else "N/A",

        "genres": genres,

        "explanation": explanation

    }

    return movie_cache[movie_title]

def get_trending_movies():

    trending_url = (
        f"https://api.themoviedb.org/3/trending/movie/day"
        f"?api_key={TMDB_API_KEY}"
    )

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        trending_url,
        headers=headers,
        timeout=10
    )

    data = response.json()

    results = data.get("results", [])

    trending_movies = []

    for movie in results[:5]:

        movie_title = movie.get("title")

        if movie_title:

            movie_data =fetch_movie_details(movie_title)

            trending_movies.append(movie_data)

    return trending_movies

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

vectors = vectorizer.fit_transform(movies['tags'])

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
        movie_data["genres"] = movie_genres.split(", ")
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

    for movie in mood_movies['title'].head(8):

        movie_data = fetch_movie_details(movie)

        movie_genres = (
            movies[movies['title'] == movie]['genres']
            .values[0]
        )

        movie_data["genres"] = movie_genres.split(", ")

        movie_data["explanation"] = (
            f"Recommended because it shares "
            f"{movie_genres} themes."
        )

        recommendations.append(movie_data)

    return recommendations