import pandas as pd
import requests
import nltk

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


API_KEY = "73f2c58b"

ps = PorterStemmer()


def stem(text):

    y = []

    for word in text.split():

        y.append(ps.stem(word))

    return " ".join(y)


def fetch_movie_details(movie_title):

    url = f"http://www.omdbapi.com/?t={movie_title}&apikey={API_KEY}"

    response = requests.get(url)

    data = response.json()

    return {
        "title": data.get("Title"),
        "poster": data.get("Poster"),
        "rating": data.get("imdbRating")
    }


movies = pd.read_csv('movies.csv')
movies = movies.head(2000)

movies = movies[
    ['title', 'overview', 'genres']
]

movies.dropna(inplace=True)

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

        movie_data = fetch_movie_details(movie_title)

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

        recommendations.append(movie_data)

    return recommendations