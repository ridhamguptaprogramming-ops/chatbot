import os

from flask import Flask, jsonify, render_template, request

from chatbot import TodoChatbot


app = Flask(__name__)
bot = TodoChatbot()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.get("/")
def home() -> str:
    return render_template("index.html")


@app.route("/api/chat", methods=["GET", "POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    if request.method == "GET":
        message = request.args.get("message", "").strip()
    else:
        data = request.get_json(silent=True) or {}
        message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Message is required."}), 400

    reply = bot.reply(message)
    return jsonify({"reply": reply})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
