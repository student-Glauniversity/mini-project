

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


data = pd.read_csv("dataset/emails.csv")

X = data['email_text']
y = data['category']


vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)


model = MultinomialNB()
model.fit(X_vec, y)


with open("models/email_classifier.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

print("Email classification model trained & saved")
