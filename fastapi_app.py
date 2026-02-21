from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from advanced_ai import AdvancedAIEngine
from chatbot import TodoChatbot


app = FastAPI(title="Todo Chatbot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = TodoChatbot()
ai_engine = AdvancedAIEngine()
root_dir = Path(__file__).resolve().parent
template_file = root_dir / "templates" / "index.html"


def _json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse({"ok": False, "error": message}, status_code=status_code)


def _parse_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return {}


def _chat_payload(message: str, session_id: str, language: str) -> dict[str, Any]:
    reply = bot.reply(message, session_id=session_id, language=language)
    return {
        "reply": reply,
        "session_id": session_id,
        "language": language,
        "state": bot.get_state(),
    }


@app.get("/")
async def home():
    if template_file.exists():
        return FileResponse(template_file)
    return _json_error("templates/index.html not found.", status_code=404)


@app.get("/health")
async def health():
    return {"ok": True, "framework": "fastapi"}


@app.get("/api/chat")
async def chat_get(
    message: str = "",
    session_id: str = "default",
    language: str = "English",
):
    clean_message = (message or "").strip()
    clean_session = (session_id or "default").strip() or "default"
    clean_language = (language or "English").strip() or "English"
    if not clean_message:
        return _json_error("Message is required.")
    return _chat_payload(clean_message, clean_session, clean_language)


@app.post("/api/chat")
async def chat_post(request: Request):
    try:
        payload = _parse_payload(await request.json())
    except Exception:
        payload = {}
    message = str(payload.get("message", "")).strip()
    session_id = str(payload.get("session_id", "default")).strip() or "default"
    language = str(payload.get("language", "English")).strip() or "English"
    if not message:
        return _json_error("Message is required.")
    return _chat_payload(message, session_id, language)


@app.get("/api/state")
async def state():
    return bot.get_state()


@app.get("/api/nlp")
async def nlp_get(
    text: str = "",
    session_id: str = "default",
    language: str = "English",
):
    clean_text = (text or "").strip()
    if not clean_text:
        return _json_error("Text is required.")
    clean_session = (session_id or "default").strip() or "default"
    clean_language = (language or "English").strip() or "English"
    return bot.nlp_analyze(text=clean_text, session_id=clean_session, language=clean_language)


@app.post("/api/nlp")
async def nlp_post(request: Request):
    try:
        payload = _parse_payload(await request.json())
    except Exception:
        payload = {}
    text = str(payload.get("text", payload.get("message", ""))).strip()
    session_id = str(payload.get("session_id", "default")).strip() or "default"
    language = str(payload.get("language", "English")).strip() or "English"
    if not text:
        return _json_error("Text is required.")
    return bot.nlp_analyze(text=text, session_id=session_id, language=language)


@app.get("/api/ai/status")
async def ai_status():
    return ai_engine.get_status()


@app.post("/api/ai/enroll-face")
async def ai_enroll_face(request: Request):
    try:
        payload = _parse_payload(await request.json())
    except Exception:
        payload = {}
    user_id = str(payload.get("user_id", "default")).strip() or "default"
    image_data = str(payload.get("image_data", "")).strip()
    if not image_data:
        return _json_error("image_data is required.")
    result = ai_engine.enroll_face(user_id=user_id, image_data=image_data)
    status_code = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status_code)


@app.post("/api/ai/verify-face")
async def ai_verify_face(request: Request):
    try:
        payload = _parse_payload(await request.json())
    except Exception:
        payload = {}
    user_id = str(payload.get("user_id", "default")).strip() or "default"
    image_data = str(payload.get("image_data", "")).strip()
    if not image_data:
        return _json_error("image_data is required.")
    result = ai_engine.verify_face(user_id=user_id, image_data=image_data)
    status_code = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status_code)


@app.post("/api/ai/emotion")
async def ai_emotion(request: Request):
    try:
        payload = _parse_payload(await request.json())
    except Exception:
        payload = {}
    text = str(payload.get("text", "")).strip()
    image_data = str(payload.get("image_data", "")).strip()
    return ai_engine.detect_emotion(text=text, image_data=image_data)


@app.post("/api/ai/gesture")
async def ai_gesture(request: Request):
    try:
        payload = _parse_payload(await request.json())
    except Exception:
        payload = {}
    image_data = str(payload.get("image_data", "")).strip()
    auto_screenshot = bool(payload.get("auto_screenshot", False))
    if not image_data:
        return _json_error("image_data is required.")
    result = ai_engine.detect_gesture_command(
        image_data=image_data,
        auto_screenshot=auto_screenshot,
    )
    status_code = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status_code)


@app.post("/api/ai/screenshot")
async def ai_screenshot(request: Request):
    try:
        payload = _parse_payload(await request.json())
    except Exception:
        payload = {}
    image_data = str(payload.get("image_data", "")).strip()
    if not image_data:
        return _json_error("image_data is required.")
    result = ai_engine.capture_screenshot(image_data=image_data)
    status_code = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status_code)


@app.post("/api/ai/automate-web")
async def ai_automate_web(request: Request):
    try:
        payload = _parse_payload(await request.json())
    except Exception:
        payload = {}
    query = str(payload.get("query", "")).strip()
    url = str(payload.get("url", "")).strip()
    result = ai_engine.automate_web(query=query, url=url)
    status_code = 200 if result.get("ok") else 400
    return JSONResponse(result, status_code=status_code)
