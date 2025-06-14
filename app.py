from flask import Flask, request, jsonify, render_template
import pickle
import os
import re
import nltk
import string
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# DO NOT download in production — do it in Dockerfile!
# nltk.download('stopwords')  <-- this line is removed

# Load model and vectorizer
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("text_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# TweetCleaner class
class TweetCleaner:
    def __init__(self):
        self.tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)
        self.stemmer = PorterStemmer()
        self.negation_words = {
            'no', 'not', 'never', 'none', "n't", 'cannot', 'cant',
            'neither', 'nor', 'nothing', 'nowhere', 'nobody', 'hardly',
            'barely', 'scarcely', 'rarely'
        }
        self.stop_words = set(stopwords.words('english')) - self.negation_words

    def remove_urls(self, text):
        return re.sub(r'https?://\S+|www\.\S+', '', text)

    def remove_mentions(self, text):
        return re.sub(r'@\w+', '', text)

    def clean_hashtags(self, text):
        return re.sub(r'#(\w+)', r'\1', text)

    def remove_punctuation_and_numbers(self, text):
        text = re.sub(r'\d+', '', text)
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_stopwords(self, text):
        tokens = self.tokenizer.tokenize(text)
        filtered = [self.stemmer.stem(word) for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered)

    def handle_negation(self, text):
        words = text.split()
        modified = []
        negation = False
        for i, word in enumerate(words):
            if word in self.negation_words or word.endswith("n't"):
                negation = True
                modified.append(word)
                continue
            if any(p in word for p in '.!?;:') or (negation and i > 2):
                negation = False
            if negation:
                modified.append('NEG_' + word)
            else:
                modified.append(word)
        return ' '.join(modified)

    def clean_tweet(self, tweet):
        tweet = tweet.lower()
        tweet = self.remove_urls(tweet)
        tweet = self.remove_mentions(tweet)
        tweet = self.clean_hashtags(tweet)
        tweet = self.remove_punctuation_and_numbers(tweet)
        tweet = self.remove_stopwords(tweet)
        tweet = self.handle_negation(tweet)
        return tweet

# Instantiate cleaner
cleaner = TweetCleaner()

# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # Make sure /templates/index.html exists

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("text", "")

    if not input_text.strip():
        return jsonify({"error": "Empty input"}), 400

    cleaned = cleaner.clean_tweet(input_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    return jsonify({"prediction": prediction})

# For local dev only — Render uses Gunicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
