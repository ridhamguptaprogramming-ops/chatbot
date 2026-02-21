import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote, quote_plus, urlparse

import requests

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None


class AdvancedAIEngine:
    def __init__(self) -> None:
        self.base_dir = Path("data/ai")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.screens_dir = self.base_dir / "screenshots"
        self.screens_dir.mkdir(parents=True, exist_ok=True)
        self.face_store_path = self.base_dir / "face_profiles.json"
        self.web_timeout = 10
        self.web_headers = {"User-Agent": "TodoChatbot-AI/1.0 (+http://127.0.0.1)"}
        self.face_threshold = 0.85
        self.face_profiles = self._load_face_profiles()
        self.face_cascade = None
        self.smile_cascade = None
        if cv2 is not None:
            self.face_cascade = cv2.CascadeClassifier(  # type: ignore[union-attr]
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[union-attr]
            )
            self.smile_cascade = cv2.CascadeClassifier(  # type: ignore[union-attr]
                cv2.data.haarcascades + "haarcascade_smile.xml"  # type: ignore[union-attr]
            )

    def get_status(self) -> dict[str, object]:
        return {
            "ok": True,
            "opencv_available": cv2 is not None and np is not None,
            "bs4_available": BeautifulSoup is not None,
            "face_profiles": len(self.face_profiles),
            "features": [
                "face_login",
                "gesture_command",
                "emotion_detection",
                "auto_screenshot",
                "smart_web_automation",
            ],
        }

    def enroll_face(self, user_id: str, image_data: str) -> dict[str, object]:
        ready = self._cv_ready()
        if ready is not None:
            return ready

        frame = self._decode_image(image_data)
        if frame is None:
            return {"ok": False, "error": "Invalid image data."}

        vector = self._extract_face_vector(frame)
        if vector is None:
            return {"ok": False, "error": "No clear face detected. Face the camera directly."}

        user_key = user_id.strip() or "default"
        self.face_profiles[user_key] = {"vector": vector, "updated_at": self._now()}
        self._save_face_profiles()
        return {
            "ok": True,
            "message": f"Face enrolled for '{user_key}'.",
            "user_id": user_key,
        }

    def verify_face(self, user_id: str, image_data: str) -> dict[str, object]:
        ready = self._cv_ready()
        if ready is not None:
            return ready

        user_key = user_id.strip() or "default"
        record = self.face_profiles.get(user_key)
        if not record:
            return {"ok": False, "error": f"No enrolled face found for '{user_key}'."}

        frame = self._decode_image(image_data)
        if frame is None:
            return {"ok": False, "error": "Invalid image data."}

        vector = self._extract_face_vector(frame)
        if vector is None:
            return {"ok": False, "error": "No clear face detected for verification."}

        stored = record.get("vector", [])
        score = self._cosine_similarity(stored, vector)
        verified = score >= self.face_threshold
        return {
            "ok": True,
            "verified": verified,
            "score": round(score, 4),
            "threshold": self.face_threshold,
            "message": "Face verified successfully." if verified else "Face did not match enrolled profile.",
        }

    def detect_emotion(self, text: str = "", image_data: str = "") -> dict[str, object]:
        text_emotion = self._detect_text_emotion(text)
        image_emotion = None
        image_confidence = 0.0

        if image_data:
            image_result = self._detect_face_emotion(image_data)
            if image_result is not None:
                image_emotion = image_result["emotion"]
                image_confidence = image_result["confidence"]

        emotion = image_emotion or text_emotion["emotion"]
        confidence = max(float(text_emotion["confidence"]), float(image_confidence))
        source = "image+text" if image_emotion else "text"
        return {
            "ok": True,
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "source": source,
            "hint": self._emotion_hint(emotion),
        }

    def detect_gesture_command(
        self,
        image_data: str,
        auto_screenshot: bool = False,
    ) -> dict[str, object]:
        ready = self._cv_ready()
        if ready is not None:
            return ready

        frame = self._decode_image(image_data)
        if frame is None:
            return {"ok": False, "error": "Invalid image data."}

        fingers = self._estimate_finger_count(frame)
        gesture, command = self._map_fingers_to_command(fingers)
        screenshot_path = None
        if auto_screenshot and fingers >= 4:
            screenshot_path = self._save_frame(frame, prefix="gesture")

        return {
            "ok": True,
            "fingers": fingers,
            "gesture": gesture,
            "command": command,
            "auto_screenshot": screenshot_path,
            "message": f"Detected {gesture}. Suggested command: {command}",
        }

    def capture_screenshot(self, image_data: str) -> dict[str, object]:
        ready = self._cv_ready()
        if ready is not None:
            return ready

        frame = self._decode_image(image_data)
        if frame is None:
            return {"ok": False, "error": "Invalid image data."}

        path = self._save_frame(frame, prefix="manual")
        return {"ok": True, "path": path, "message": "Screenshot captured."}

    def automate_web(self, query: str = "", url: str = "") -> dict[str, object]:
        query_value = query.strip()
        url_value = url.strip()

        if query_value:
            search_payload = self._search_web(query_value)
            if search_payload:
                search_payload["ok"] = True
                return search_payload
            return {
                "ok": False,
                "error": "No live web result found.",
                "google_url": self._google_link(query_value),
            }

        if url_value:
            return self._summarize_page(url_value)

        return {"ok": False, "error": "Provide either query or url for web automation."}

    def _search_web(self, query: str) -> Optional[dict[str, object]]:
        payload = self._search_duckduckgo(query)
        if payload:
            return payload
        return self._search_wikipedia(query)

    def _search_duckduckgo(self, query: str) -> Optional[dict[str, object]]:
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
            "no_redirect": "1",
        }
        try:
            response = requests.get(
                "https://api.duckduckgo.com/",
                params=params,
                timeout=self.web_timeout,
                headers=self.web_headers,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            return None

        summary = str(data.get("AbstractText") or data.get("Answer") or "").strip()
        results: list[dict[str, str]] = []
        seen: set[str] = set()

        abstract_url = str(data.get("AbstractURL") or "").strip()
        if abstract_url:
            seen.add(abstract_url)
            results.append(
                {
                    "title": str(data.get("Heading") or query).strip(),
                    "url": abstract_url,
                    "snippet": summary,
                }
            )

        for item in self._flatten_topics(data.get("RelatedTopics", [])):
            url_value = item.get("url", "").strip()
            text_value = item.get("text", "").strip()
            if not url_value or not text_value or url_value in seen:
                continue
            seen.add(url_value)
            results.append(
                {
                    "title": text_value.split(" - ", 1)[0][:90],
                    "url": url_value,
                    "snippet": text_value,
                }
            )
            if len(results) >= 5:
                break

        if not summary and not results:
            return None

        return {
            "mode": "search",
            "source": "DuckDuckGo",
            "query": query,
            "summary": self._shorten(summary or (results[0]["snippet"] if results else ""), 220),
            "results": results[:5],
            "google_url": self._google_link(query),
        }

    def _search_wikipedia(self, query: str) -> Optional[dict[str, object]]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "utf8": "1",
            "format": "json",
            "srlimit": "5",
        }
        try:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                timeout=self.web_timeout,
                headers=self.web_headers,
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            return None

        rows = data.get("query", {}).get("search", [])
        if not isinstance(rows, list) or not rows:
            return None

        results: list[dict[str, str]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            if not title:
                continue
            snippet = self._strip_html(str(row.get("snippet") or ""))
            results.append(
                {
                    "title": title,
                    "url": "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_")),
                    "snippet": snippet,
                }
            )
            if len(results) >= 5:
                break

        if not results:
            return None

        return {
            "mode": "search",
            "source": "Wikipedia",
            "query": query,
            "summary": self._shorten(results[0]["snippet"], 220),
            "results": results,
            "google_url": self._google_link(query),
        }

    def _summarize_page(self, raw_url: str) -> dict[str, object]:
        parsed = urlparse(raw_url)
        url_value = raw_url
        if not parsed.scheme:
            url_value = f"https://{raw_url}"

        try:
            response = requests.get(url_value, timeout=self.web_timeout, headers=self.web_headers)
            response.raise_for_status()
            html = response.text
        except requests.RequestException as exc:
            return {"ok": False, "error": f"Failed to open URL: {exc}"}

        if BeautifulSoup is None:
            title = self._extract_html_title(html)
            return {
                "ok": True,
                "mode": "page",
                "url": url_value,
                "title": title or "Page loaded",
                "description": "",
                "top_links": [],
            }

        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.string.strip() if soup.title and soup.title.string else "") or "Untitled"
        description = ""
        desc_tag = soup.find("meta", attrs={"name": "description"})
        if desc_tag and isinstance(desc_tag.get("content"), str):
            description = desc_tag["content"].strip()

        top_links: list[dict[str, str]] = []
        seen: set[str] = set()
        for tag in soup.find_all("a", href=True):
            href = str(tag.get("href") or "").strip()
            text = " ".join(tag.get_text(" ", strip=True).split())
            if not href or href.startswith("#"):
                continue
            if href.startswith("/"):
                href = url_value.rstrip("/") + href
            if not href.startswith("http"):
                continue
            if href in seen:
                continue
            seen.add(href)
            top_links.append({"title": text[:80] or "link", "url": href})
            if len(top_links) >= 5:
                break

        return {
            "ok": True,
            "mode": "page",
            "url": url_value,
            "title": title,
            "description": self._shorten(description, 220),
            "top_links": top_links,
        }

    def _load_face_profiles(self) -> dict[str, dict[str, object]]:
        if not self.face_store_path.exists():
            return {}
        try:
            raw = json.loads(self.face_store_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        clean: dict[str, dict[str, object]] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            vector = value.get("vector")
            if not isinstance(vector, list):
                continue
            clean[key] = {
                "vector": [float(item) for item in vector if isinstance(item, (int, float))],
                "updated_at": str(value.get("updated_at", "")),
            }
        return clean

    def _save_face_profiles(self) -> None:
        try:
            self.face_store_path.write_text(
                json.dumps(self.face_profiles, ensure_ascii=True, indent=2),
                encoding="utf-8",
            )
        except OSError:
            return

    def _decode_image(self, image_data: str):
        if cv2 is None or np is None:
            return None
        value = (image_data or "").strip()
        if not value:
            return None
        if "," in value:
            value = value.split(",", 1)[1]
        try:
            raw = base64.b64decode(value)
        except (ValueError, TypeError):
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)  # type: ignore[union-attr]
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # type: ignore[union-attr]
        return frame

    def _extract_face_vector(self, frame):
        if cv2 is None or np is None or self.face_cascade is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore[union-attr]
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        roi = gray[y : y + h, x : x + w]
        roi = cv2.resize(roi, (128, 128))  # type: ignore[union-attr]
        hist = cv2.calcHist([roi], [0], None, [64], [0, 256])  # type: ignore[union-attr]
        hist = cv2.normalize(hist, hist).flatten()  # type: ignore[union-attr]
        return [float(item) for item in hist]

    def _detect_face_emotion(self, image_data: str) -> Optional[dict[str, object]]:
        if cv2 is None or np is None or self.face_cascade is None:
            return None
        frame = self._decode_image(image_data)
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type: ignore[union-attr]
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=6, minSize=(80, 80)
        )
        if len(faces) == 0:
            return None

        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        roi = gray[y : y + h, x : x + w]

        happy_conf = 0.0
        if self.smile_cascade is not None:
            smiles = self.smile_cascade.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=22)
            happy_conf = min(1.0, 0.25 * len(smiles))

        brightness = float(np.mean(roi)) if np is not None else 0.0
        if happy_conf >= 0.5:
            return {"emotion": "happy", "confidence": max(0.7, happy_conf)}
        if brightness < 65:
            return {"emotion": "sad", "confidence": 0.58}
        if brightness > 165:
            return {"emotion": "excited", "confidence": 0.55}
        return {"emotion": "neutral", "confidence": 0.62}

    def _detect_text_emotion(self, text: str) -> dict[str, object]:
        value = (text or "").strip().lower()
        if not value:
            return {"emotion": "neutral", "confidence": 0.5}

        positive = {
            "good",
            "great",
            "awesome",
            "happy",
            "love",
            "excited",
            "amazing",
            "nice",
        }
        negative = {
            "bad",
            "sad",
            "angry",
            "upset",
            "stressed",
            "depressed",
            "hate",
            "tired",
        }
        calm = {"okay", "fine", "normal", "alright"}

        words = re.findall(r"[a-z']+", value)
        score = sum(1 for item in words if item in positive) - sum(
            1 for item in words if item in negative
        )

        if score >= 2:
            return {"emotion": "happy", "confidence": 0.76}
        if score <= -2:
            return {"emotion": "sad", "confidence": 0.76}
        if any(item in words for item in negative):
            return {"emotion": "stressed", "confidence": 0.64}
        if any(item in words for item in positive):
            return {"emotion": "positive", "confidence": 0.64}
        if any(item in words for item in calm):
            return {"emotion": "neutral", "confidence": 0.6}
        return {"emotion": "neutral", "confidence": 0.52}

    def _emotion_hint(self, emotion: str) -> str:
        hints = {
            "happy": "Great mood. Keep momentum and plan next task.",
            "excited": "High energy detected. Good time for priority tasks.",
            "sad": "Low mood detected. Start with a small, easy task first.",
            "stressed": "Take a short break and then do one task at a time.",
            "positive": "Positive tone detected. Keep going.",
            "neutral": "Balanced mood. Continue current workflow.",
        }
        return hints.get(emotion, "Mood processed.")

    def _estimate_finger_count(self, frame) -> int:
        if cv2 is None or np is None:
            return 0
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)  # type: ignore[union-attr]
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  # type: ignore[union-attr]

        lower_skin = np.array([0, 20, 40], dtype=np.uint8)  # type: ignore[union-attr]
        upper_skin = np.array([30, 255, 255], dtype=np.uint8)  # type: ignore[union-attr]
        mask = cv2.inRange(hsv, lower_skin, upper_skin)  # type: ignore[union-attr]
        kernel = np.ones((3, 3), dtype=np.uint8)  # type: ignore[union-attr]
        mask = cv2.erode(mask, kernel, iterations=1)  # type: ignore[union-attr]
        mask = cv2.dilate(mask, kernel, iterations=2)  # type: ignore[union-attr]
        mask = cv2.GaussianBlur(mask, (5, 5), 0)  # type: ignore[union-attr]

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # type: ignore[union-attr]
        if not contours:
            return 0
        cnt = max(contours, key=cv2.contourArea)  # type: ignore[union-attr]
        area = cv2.contourArea(cnt)  # type: ignore[union-attr]
        if area < 4000:
            return 0

        hull = cv2.convexHull(cnt, returnPoints=False)  # type: ignore[union-attr]
        if hull is None or len(hull) < 4:
            return 1
        defects = cv2.convexityDefects(cnt, hull)  # type: ignore[union-attr]
        if defects is None:
            return 1

        defect_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = cnt[s][0]
            end = cnt[e][0]
            far = cnt[f][0]

            a = np.linalg.norm(end - start)  # type: ignore[union-attr]
            b = np.linalg.norm(far - start)  # type: ignore[union-attr]
            c = np.linalg.norm(end - far)  # type: ignore[union-attr]
            if b * c == 0:
                continue
            angle = np.degrees(np.arccos((b * b + c * c - a * a) / (2 * b * c)))  # type: ignore[union-attr]
            if angle <= 90 and d > 10000:
                defect_count += 1

        return int(max(0, min(5, defect_count + 1)))

    def _map_fingers_to_command(self, fingers: int) -> tuple[str, str]:
        if fingers <= 0:
            return "fist", "help"
        if fingers == 1:
            return "one_finger", "list open"
        if fingers == 2:
            return "two_fingers", "stats"
        if fingers == 3:
            return "three_fingers", "done 1"
        if fingers == 4:
            return "four_fingers", "search task"
        return "open_palm", "clear done"

    def _save_frame(self, frame, prefix: str) -> str:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        path = self.screens_dir / f"{prefix}_{stamp}.jpg"
        cv2.imwrite(str(path), frame)  # type: ignore[union-attr]
        return str(path)

    def _flatten_topics(self, topics: object) -> list[dict[str, str]]:
        if not isinstance(topics, list):
            return []
        rows: list[dict[str, str]] = []
        for item in topics:
            if not isinstance(item, dict):
                continue
            first_url = item.get("FirstURL")
            text = item.get("Text")
            if isinstance(first_url, str) and isinstance(text, str):
                rows.append({"url": first_url, "text": text})
            nested = item.get("Topics")
            if isinstance(nested, list):
                rows.extend(self._flatten_topics(nested))
        return rows

    def _extract_html_title(self, html: str) -> str:
        match = re.search(r"<title>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return re.sub(r"\s+", " ", match.group(1)).strip()

    def _strip_html(self, text: str) -> str:
        return re.sub(r"<[^>]+>", "", text).strip()

    def _shorten(self, text: str, limit: int) -> str:
        value = " ".join((text or "").split())
        if len(value) <= limit:
            return value
        return value[: limit - 1].rstrip() + "..."

    def _cosine_similarity(self, left: list[float], right: list[float]) -> float:
        if np is None:
            return 0.0
        if not left or not right:
            return 0.0
        a = np.array(left, dtype=np.float32)  # type: ignore[union-attr]
        b = np.array(right, dtype=np.float32)  # type: ignore[union-attr]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))  # type: ignore[union-attr]
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)  # type: ignore[union-attr]

    def _google_link(self, query: str) -> str:
        return f"https://www.google.com/search?q={quote_plus(query)}"

    def _cv_ready(self) -> Optional[dict[str, object]]:
        if cv2 is None or np is None:
            return {"ok": False, "error": "OpenCV/Numpy not installed in Python environment."}
        return None

    def _now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"
