from flask import Flask, render_template, request, jsonify
import nltk
from nltk.chat.util import Chat, reflections
from pairs import pairs
import random

# Initialize Flask
app = Flask(__name__)

# Initialize chatbot
chatbot = Chat(pairs, reflections)

# Default responses for unrecognized inputs
default_responses = [
    "I'm sorry, I didn't understand that. Can you rephrase?",
    "Could you clarify your question, please?",
    "I'm not sure how to respond to that. Try asking something else about the hotel."
]

# Route for homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle chatbot response
@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.json.get("message", "").strip()

    if not user_input:
        return jsonify({"response": "Please enter a valid query."})

    response = chatbot.respond(user_input.lower())

    if not response:
        response = random.choice(default_responses)

    return jsonify({"response": response})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
