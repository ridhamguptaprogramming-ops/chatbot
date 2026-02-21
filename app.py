import os

from flask import Flask, jsonify, render_template, request

from chatbot import TodoChatbot


app = Flask(__name__)
bot = TodoChatbot()


@app.get("/")
def home() -> str:
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is required."}), 400

    reply = bot.reply(message)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
