from flask import Flask, render_template, request
import openai

# ✅ RECOMMENDED: Store your key securely via environment variable or config
# For now, you can paste it here for local testing:
# Replace with your API Key
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get")
def get_response():
    user_input = request.args.get("msg")

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=100,
            temperature=0.7,
        )
        bot_message = response["choices"][0]["message"]["content"].strip()
        return bot_message
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
