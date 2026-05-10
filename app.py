from flask import Flask, render_template, request
from recommendation import recommend, recommend_by_mood

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():

    recommendations = []

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
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)
