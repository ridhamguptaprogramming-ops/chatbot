import os
import re
from typing import Optional

import requests


class TodoChatbot:
    def __init__(self) -> None:
        self.todos: list[dict[str, object]] = []
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        self.system_prompt = (
            "You are a concise, helpful assistant for a todo app. "
            "Give practical answers and keep responses short."
        )

    def reply(self, message: str) -> str:
        local_result = self._handle_todo_command(message)
        if local_result is not None:
            return local_result

        if self.api_key:
            llm_response = self._ask_openai(message)
            if llm_response:
                return llm_response

        return self._fallback_response(message)

    def _handle_todo_command(self, message: str) -> Optional[str]:
        text = message.strip()
        lower = text.lower()

        if lower in {"help", "commands"}:
            return (
                "Commands:\n"
                "- add <task>\n"
                "- list\n"
                "- done <number>\n"
                "- delete <number>"
            )

        add_match = re.match(r"^(add|new|todo)\s+(.+)$", text, flags=re.IGNORECASE)
        if add_match:
            task = add_match.group(2).strip()
            self.todos.append({"text": task, "done": False})
            return f"Added task #{len(self.todos)}: {task}"

        if lower in {"list", "tasks", "todos", "show tasks", "show todos"}:
            if not self.todos:
                return "No tasks yet. Add one with: add <task>"

            lines = []
            for idx, item in enumerate(self.todos, start=1):
                status = "x" if bool(item["done"]) else " "
                lines.append(f"{idx}. [{status}] {item['text']}")
            return "\n".join(lines)

        done_match = re.match(r"^(done|complete|finish)\s+(\d+)$", lower)
        if done_match:
            idx = int(done_match.group(2))
            if idx < 1 or idx > len(self.todos):
                return "Invalid task number."
            self.todos[idx - 1]["done"] = True
            return f"Marked task #{idx} as done."

        delete_match = re.match(r"^(delete|remove)\s+(\d+)$", lower)
        if delete_match:
            idx = int(delete_match.group(2))
            if idx < 1 or idx > len(self.todos):
                return "Invalid task number."
            removed = self.todos.pop(idx - 1)
            return f"Deleted task #{idx}: {removed['text']}"

        return None

    def _ask_openai(self, message: str) -> Optional[str]:
        url = "https://api.openai.com/v1/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}],
                },
            ],
            "max_output_tokens": 300,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException:
            return None

        text = (data.get("output_text") or "").strip()
        if text:
            return text

        output = data.get("output", [])
        for item in output:
            for chunk in item.get("content", []):
                if chunk.get("type") == "output_text":
                    candidate = (chunk.get("text") or "").strip()
                    if candidate:
                        return candidate
        return None

    def _fallback_response(self, message: str) -> str:
        lower = message.lower().strip()

        if any(greeting in lower for greeting in ("hi", "hello", "hey")):
            return "Hi! I can manage todos. Try: add buy milk"
        if "name" in lower:
            return "I am your Todo Chatbot."
        if "how are you" in lower:
            return "Doing well. Want to add a task?"

        return (
            "I can help with todos right now. Use:\n"
            "- add <task>\n"
            "- list\n"
            "- done <number>\n"
            "- delete <number>"
        )
