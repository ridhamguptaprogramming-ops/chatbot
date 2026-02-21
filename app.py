import os

from flask import Flask, jsonify, render_template, request # type: ignore

from advanced_ai import AdvancedAIEngine
from chatbot import TodoChatbot


app = Flask(__name__)
bot = TodoChatbot()
ai_engine = AdvancedAIEngine()


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


@app.route("/api/nlp", methods=["GET", "POST", "OPTIONS"])
def nlp():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})

    if request.method == "GET":
        text = str(request.args.get("text", "")).strip()
        session_id = str(request.args.get("session_id", "default")).strip() or "default"
        language = str(request.args.get("language", "English")).strip() or "English"
    else:
        data = request.get_json(silent=True) or {}
        text = str(data.get("text", data.get("message", ""))).strip()
        session_id = str(data.get("session_id", "default")).strip() or "default"
        language = str(data.get("language", "English")).strip() or "English"

    if not text:
        return jsonify({"ok": False, "error": "Text is required."}), 400

    return jsonify(bot.nlp_analyze(text=text, session_id=session_id, language=language))


@app.route("/api/ai/status", methods=["GET", "OPTIONS"])
def ai_status():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    return jsonify(ai_engine.get_status())


@app.route("/api/ai/enroll-face", methods=["POST", "OPTIONS"])
def ai_enroll_face():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "default")).strip() or "default"
    image_data = str(data.get("image_data", "")).strip()
    if not image_data:
        return jsonify({"ok": False, "error": "image_data is required."}), 400
    result = ai_engine.enroll_face(user_id=user_id, image_data=image_data)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/ai/verify-face", methods=["POST", "OPTIONS"])
def ai_verify_face():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    data = request.get_json(silent=True) or {}
    user_id = str(data.get("user_id", "default")).strip() or "default"
    image_data = str(data.get("image_data", "")).strip()
    if not image_data:
        return jsonify({"ok": False, "error": "image_data is required."}), 400
    result = ai_engine.verify_face(user_id=user_id, image_data=image_data)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/ai/emotion", methods=["POST", "OPTIONS"])
def ai_emotion():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()
    image_data = str(data.get("image_data", "")).strip()
    result = ai_engine.detect_emotion(text=text, image_data=image_data)
    return jsonify(result)


@app.route("/api/ai/gesture", methods=["POST", "OPTIONS"])
def ai_gesture():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    data = request.get_json(silent=True) or {}
    image_data = str(data.get("image_data", "")).strip()
    auto_screenshot = bool(data.get("auto_screenshot", False))
    if not image_data:
        return jsonify({"ok": False, "error": "image_data is required."}), 400
    result = ai_engine.detect_gesture_command(
        image_data=image_data,
        auto_screenshot=auto_screenshot,
    )
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/ai/screenshot", methods=["POST", "OPTIONS"])
def ai_screenshot():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    data = request.get_json(silent=True) or {}
    image_data = str(data.get("image_data", "")).strip()
    if not image_data:
        return jsonify({"ok": False, "error": "image_data is required."}), 400
    result = ai_engine.capture_screenshot(image_data=image_data)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


@app.route("/api/ai/automate-web", methods=["POST", "OPTIONS"])
def ai_automate_web():
    if request.method == "OPTIONS":
        return jsonify({"ok": True})
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    url = str(data.get("url", "")).strip()
    result = ai_engine.automate_web(query=query, url=url)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
