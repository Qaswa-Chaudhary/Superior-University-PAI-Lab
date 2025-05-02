from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- NLTK setup ---
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# --- Load model and tokenizer ---
model = tf.keras.models.load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# --- Flask App ---
app = Flask(__name__)

# --- Preprocessing Parameters ---
max_len = 200  # Must match training config
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# --- Preprocessing Function ---
def text_preprocessing(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# --- Prediction Route ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    
    # Preprocess the text
    clean_text = text_preprocessing(text)
    
    # Tokenize and pad
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded)[0][0]
    sentiment = "positive" if prediction >= 0.5 else "negative"
    
    return jsonify({
        'text': text,
        'cleaned_text': clean_text,
        'sentiment': sentiment,
        'confidence': float(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True)
