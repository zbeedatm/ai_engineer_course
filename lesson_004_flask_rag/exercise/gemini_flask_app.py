import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from google import genai

load_dotenv()

# ---------------------------------------
# Gemini Client Setup
# ---------------------------------------
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ---------------------------------------
# Flask App
# ---------------------------------------
app = Flask(__name__)

def ask_gemini(message: str) -> str:
    """Send a user message to Gemini and return the model's reply."""
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=message
    )
    return response.text


@app.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint: expects JSON { 'message': '...' }"""
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    user_message = data["message"]
    reply = ask_gemini(user_message)

    return jsonify({"reply": reply})


# ---------------------------------------
# Run the server
# ---------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
