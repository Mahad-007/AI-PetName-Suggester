from flask import Flask, request, jsonify, render_template
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

with open('pet_names.json', 'r') as file:
    data = json.load(file)

df = pd.DataFrame(data)
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['trait'])
y = df['name']
model = LogisticRegression()
model.fit(X, y)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/suggest', methods=['POST'])
def suggest_name():
    req = request.json
    pet_type = req.get('type', '')
    trait = req.get('trait', '')

    input_text = f"{pet_type} {trait}"
    pet_vector = tfidf.transform([input_text])
    suggested_name = model.predict(pet_vector)

    return jsonify({"suggested_name": suggested_name[0]})


if __name__ == '__main__':
    app.run()
