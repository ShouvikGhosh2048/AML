from flask import Flask, request
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from score import score

# Model used for testing.
train_x = []
train_y = []
with open('../Assignment2/train.csv', newline='') as f:
    reader = csv.reader(f, delimiter=',', quotechar='|')
    for [line, label] in reader:
        train_x.append(line)
        train_y.append(int(label))

vectorizer = TfidfVectorizer()
train_x = vectorizer.fit_transform(train_x)

model_logistic_regression = LogisticRegression(random_state=0)
model_logistic_regression.fit(train_x, train_y)

app = Flask(__name__)

@app.post("/score")
def score_post():
    (prediction, propensity) = score(request.form['text'], model_logistic_regression, 0.5)
    return {
        "prediction": prediction,
        "propensity": propensity,
    }