from flask import Flask, request, render_template, jsonify
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import contractions

nltk.download('punkt')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
from chat import get_response  # Assuming you have a chat module
# Load the model and tokenizer
model = tf.keras.models.load_model('best_model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 100  # Define max_len used in training

# Initialize Flask app
app = Flask(__name__)


# Route for About Us page
@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')
# Preprocess function
def preprocess_text(text):
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = [w for w in text.split() if w not in stop_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return words

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for sentiment analysis service
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text')
    preprocessed_text = preprocess_text(text)
    xt = tokenizer.texts_to_sequences([text])
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    prediction = model.predict(xt).argmax(axis=1)[0]
    return jsonify({'sentiment': sentiment_classes[prediction]})

@app.route('/predict', methods=['POST'])
def predict():
    text = request.get_json().get('message')
    response = get_response(text)
    message = {'answer': response}
    return jsonify(message)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
