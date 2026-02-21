import os

from flask import Flask, jsonify, render_template, request # type: ignore

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
        session_id = str(request.args.get("session_id", "default")).strip() or "default"
        language = str(request.args.get("language", "English")).strip() or "English"
    else:
        data = request.get_json(silent=True) or {}
        message = str(data.get("message", "")).strip()
        session_id = str(data.get("session_id", "default")).strip() or "default"
        language = str(data.get("language", "English")).strip() or "English"

    if not message:
        return jsonify({"error": "Message is required."}), 400

    reply = bot.reply(message, session_id=session_id, language=language)
    return jsonify(
        {
            "reply": reply,
            "session_id": session_id,
            "language": language,
            "state": bot.get_state(),
        }
    )


@app.route("/api/state", methods=["GET", "OPTIONS"])
def state():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    return jsonify(bot.get_state())


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
