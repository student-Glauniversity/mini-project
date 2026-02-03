from flask import Flask, render_template, request
import pickle
from sentiment import get_sentiment

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

def get_priority(sentiment):
    if sentiment == "Negative":
        return "High"
    elif sentiment == "Neutral":
        return "Medium"
    return "Low"
def generate_reply(category, sentiment):
    if category == "Complaint":
        return "We are sorry for the inconvenience. Our support team is working on your issue."
    elif category == "Refund":
        return "Your refund request has been received and is currently under process."
    elif category == "Feedback":
        return "Thank you for your valuable feedback. We appreciate it."
    else:
        return "Thank you for contacting us. We will get back to you shortly."
with open("models/email_classifier.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template("login.html")

@app.route('/dashboard', methods=['POST'])
def dashboard():
    email_text = request.form['email']

    email_vec = vectorizer.transform([email_text])
    category = model.predict(email_vec)[0]

    sentiment = get_sentiment(email_text)
    priority = get_priority(sentiment)
    reply = generate_reply(category, sentiment)


    return render_template(
    "dashboard.html",
    email=email_text,
    category=category,
    sentiment=sentiment,
    priority=priority,
    reply=reply
)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

