
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import string
import os

app = Flask(__name__)
CORS(app)

# Get the backend folder location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load the trained models and vectorizer
LR = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
GB = joblib.load(os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
RF = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
vectorization = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))


# Text preprocessing (same function as your notebook)
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# Convert numerical prediction to text
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"


# Prediction function
def predict_news(news):

    cleaned_news = wordopt(news)

    # Convert text into TF-IDF features
    news_vector = vectorization.transform([cleaned_news])

    # Predictions from all 3 models
    pred_lr = int(LR.predict(news_vector)[0])
    pred_gb = int(GB.predict(news_vector)[0])
    pred_rf = int(RF.predict(news_vector)[0])

    # Convert predictions to labels
    lr_result = output_label(pred_lr)
    gb_result = output_label(pred_gb)
    rf_result = output_label(pred_rf)

    predictions = [pred_lr, pred_gb, pred_rf]

    # Count votes
    fake_votes = predictions.count(0)
    true_votes = predictions.count(1)

    # Majority verdict
    if fake_votes > true_votes:
        final_verdict = "Fake News"
    elif true_votes > fake_votes:
        final_verdict = "Not A Fake News"
    else:
        final_verdict = "Inconclusive"

    # Return results
    return {
        "logistic_regression": lr_result,
        "gradient_boosting": gb_result,
        "random_forest": rf_result,
        "final_verdict": final_verdict,
        "votes_fake": fake_votes,
        "votes_true": true_votes,
        "consensus": f"{max(fake_votes, true_votes)}/3"
    }


# Home route - check whether backend is working
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "message": "Fake News Detection API is running"
    })


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        if not data or "news" not in data:
            return jsonify({"error": "News text is missing"}), 400

        news = data["news"]

        if not isinstance(news, str) or not news.strip():
            return jsonify({"error": "News text cannot be empty"}), 400

        if len(news.strip()) < 30:
            return jsonify({"error": "Please enter a longer news article"}), 400

        result = predict_news(news)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
