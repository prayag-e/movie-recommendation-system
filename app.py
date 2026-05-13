from flask import Flask, render_template, request
from recommendation import recommend, recommend_by_mood
from flask import jsonify
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():

    trending_movies = [
    "Inception",
    "Interstellar",
    "The Dark Knight",
    "Avengers: Endgame",
    "Titanic"]

    recommendations = []

    default_movies = []

    for movie in trending_movies:

        movie_data = recommend(movie)

        if movie_data:

            default_movies.append(movie_data[0])

    error = None

    if request.method == 'POST':

        movie_name = request.form['movie']

        mood = request.form['mood']

        if movie_name.strip() != "":

            recommendations = recommend(movie_name)

            if not recommendations:

                error = "Movie not found. Try another movie."

        elif mood != "":

            recommendations = recommend_by_mood(mood)

    return render_template(
    'index.html',
    recommendations=recommendations,
    default_movies=default_movies,
    error=error
)
@app.route('/suggest')

def suggest():

    query = request.args.get('q')

    suggestions = []

    if query:

        for title in recommend.__globals__['movies']['title']:

            if query.lower() in str(title).lower():

                suggestions.append(title)

            if len(suggestions) >= 5:
                break

    return jsonify(suggestions)
if __name__ == '__main__':
    app.run(debug=True)
