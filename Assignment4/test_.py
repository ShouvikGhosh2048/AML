# I have used https://www.freecodecamp.org/news/how-to-dockerize-a-flask-app/ as a resource for assignment 4, as mentioned in the assignment.
# Test commit for hooks
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from score import score
import urllib.request
import urllib.parse
from subprocess import Popen
import time
import json

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

def test_score():
    (prediction, propensity) = score('Hi', model_logistic_regression, 0.5)
    assert type(prediction) == bool
    assert type(propensity) == float
    assert 0.0 <= propensity and propensity <= 1.0

    (prediction, _) = score("Hi! How's life?", model_logistic_regression, 0.5)
    assert not prediction

    (prediction, _) = score("BUY CARS AT A DISCOUNT. Get a free test drive today. 100 lucky customers will win prizes, like free water bottles, free bags, free shoes, free books, free laptops and various other free stuff. You can win free items too.", model_logistic_regression, 0.5)
    assert prediction

# https://www.howtogeek.com/devops/linux-signals-hacks-definition-and-more/
# https://docs.python.org/3/library/subprocess.html#popen-constructor
# https://docs.python.org/3/library/json.html?highlight=json#module-json
# https://docs.python.org/3/library/urllib.html?highlight=urllib#module-urllib
def test_flask():
    process = Popen(["py", "-m", "flask", "--app", "app", "run"])
    time.sleep(10)
    data = urllib.parse.urlencode({'text': 'Hi'})
    data = data.encode('ascii')
    with urllib.request.urlopen("http://127.0.0.1:5000/score", data) as f:
        response = f.read().decode('utf-8')
        response_json = json.loads(response)
        assert 'prediction' in response_json and 'propensity' in response_json
    process.terminate()

def test_docker():
    process = Popen(["docker", "build", "--tag", "aml_assignment_container", "."])
    process.wait()
    process = Popen(["docker", "run", "-d", "-p", "5000:5000", "--name", "aml_assignment_image", "aml_assignment_container"])
    process.wait()
    time.sleep(10)
    data = urllib.parse.urlencode({'text': 'Hi'})
    data = data.encode('ascii')
    with urllib.request.urlopen("http://127.0.0.1:5000/score", data) as f:
        response = f.read().decode('utf-8')
        response_json = json.loads(response)
        assert 'prediction' in response_json and 'propensity' in response_json
    process = Popen(["docker", "stop", "aml_assignment_image"])
    process.wait()
    process = Popen(["docker", "rm", "aml_assignment_image"])
    process.wait()
    process = Popen(["docker", "rmi", "aml_assignment_container"])
    process.wait()