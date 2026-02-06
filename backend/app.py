from flask import Flask, render_template, request
import pickle
import os

# ================== PATH SETUP ==================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "email_classifier.pkl")

print("PROJECT_ROOT:", PROJECT_ROOT)
print("TEMPLATE_DIR:", TEMPLATE_DIR)
print("STATIC_DIR:", STATIC_DIR)
print("MODEL_PATH:", MODEL_PATH)

# ================== APP ==================

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)

# ================== LOAD MODEL ==================

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

if isinstance(data, tuple):
    model, vectorizer = data
elif isinstance(data, dict):
    model = data.get("model")
    vectorizer = data.get("vectorizer")
else:
    model = data
    vectorizer = None


# ================== LOGIC ==================

def get_sentiment(text):
    text = text.lower()
    if any(word in text for word in ["angry", "bad", "worst", "refund", "complaint"]):
        return "Negative"
    elif any(word in text for word in ["good", "great", "thank", "thanks"]):
        return "Positive"
    return "Neutral"

def get_priority(sentiment):
    return "High" if sentiment == "Negative" else "Medium" if sentiment == "Neutral" else "Low"

def generate_reply(category, sentiment):
    responses = {
        "Complaint": "We are sorry for the inconvenience. Our support team is working on your issue.",
        "Refund": "Your refund request has been received and is currently under process.",
        "Feedback": "Thank you for your valuable feedback. We appreciate it."
    }
    return responses.get(category, "Thank you for contacting us. We will get back to you shortly.")

# ================== ROUTES ==================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze")
def analyze():
    return render_template("analyze.html")

@app.route("/dashboard", methods=["POST"])
def dashboard():
    email_text = request.form["email"]

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

@app.route("/login")
def login():
    return render_template("login.html")

# ================== RUN ==================

if __name__ == "__main__":
    app.run(debug=True)
