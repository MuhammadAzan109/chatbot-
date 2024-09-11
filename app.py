from flask import Flask, render_template, request, jsonify
import numpy as np
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import pickle
import random
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Load the model and data
model = load_model('chatbot_model.h5')
with open('data.pkl', 'rb') as file:
    data = pickle.load(file)

words = data['words']
classes = data['classes']
intents = json.load(open('data.json'))

lemmatizer = WordNetLemmatizer()

# Preprocess user input
def preprocess_input(user_input):
    word_list = nltk.word_tokenize(user_input)
    word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
    bag = [1 if word in word_list else 0 for word in words]
    return np.array(bag).reshape(1, -1)

# Get response from the model
def get_response(user_input):
    bag_of_words = preprocess_input(user_input)
    prediction = model.predict(bag_of_words)
    class_id = np.argmax(prediction)
    response_tag = classes[class_id]

    for intent in intents['intents']:
        if intent['tag'] == response_tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    user_input = request.form['msg']
    response = get_response(user_input)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Change to another port if needed

