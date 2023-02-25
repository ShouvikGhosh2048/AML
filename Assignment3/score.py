import csv
from sklearn.feature_extraction.text import TfidfVectorizer

train_x = []
with open('../Assignment2/train.csv', newline='') as f:
    reader = csv.reader(f, delimiter=',', quotechar='|')
    for [line, _] in reader:
        train_x.append(line)

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(train_x)

def score(text: str, model, threshold:float) -> tuple[bool, float]:
    data = [text]
    data = vectorizer.transform(data)
    spam_probability = model.predict_proba(data)[0][1]
    return (bool(spam_probability > threshold), float(spam_probability))